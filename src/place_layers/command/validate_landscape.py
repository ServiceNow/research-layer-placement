"""Validate landscape command: per-band real evaluation of surrogate predictions.

For each cost band:
1. Compute surrogate DP optimal (single Ridge MRF, like placement_landscape)
2. Run ViterbiSurrogateSearch restricted to that band (bootstrap ensemble + EI + real eval)
3. Save both surrogate DP and validated search results to JSON

Per-band results are saved incrementally under ``bands/band_NNN/`` so that a
crash mid-way loses only the current band.  On restart, completed bands are
detected and skipped automatically.

Input: exploration_results.json + evaluator config + cost model config.
Output: validation_results.json in Hydra output dir (assembled from bands/).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from hydra.core.hydra_config import HydraConfig
from pydantic import Field
from scipy.stats import spearmanr

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.landscape import (
    BandValidationResult,
    LandscapeConfig,
    PlacementResult,
    ValidationResults,
    ValidationSummary,
    compute_band_edges,
    compute_dp_per_band,
    load_and_fit_surrogate,
)
from place_layers.cost_model.enumeration import enumerate_feasible_allocations
from place_layers.backend import BackendConfig
from place_layers.evaluator.evaluators import EvaluatorsConfig
from place_layers.problem.layer_placement import (
    CostConstrainedProblem,
    MaxPerformanceProblemConfig,
)
from place_layers.search_strategy.cost_constrained.viterbi_surrogate import (
    ViterbiSurrogateSearchConfig,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Command
# ============================================================================


class ValidateLandscapeCommand(BaseCommand):
    """Validate placement landscape predictions against real evaluator."""

    config: "ValidateLandscapeCommandConfig"

    async def run(self) -> None:
        """Execute per-band validation with incremental saving."""
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)
        logger.info("Output directory: %s", output_dir)

        landscape = self.config.landscape

        # 1. Fit surrogate, compute bands (shared helpers)
        dataset, surrogate, cost_model = load_and_fit_surrogate(
            landscape.exploration_dataset_path,
            landscape.cost_model,
            landscape.seed,
        )
        num_layers = dataset.num_layers

        # Enumerate ALL allocations (no budget filter)
        all_allocations = enumerate_feasible_allocations(
            cost_model, num_layers, cost_budget=None
        )
        alloc_costs = np.array(
            [cost_model.allocation_cost(a) for a in all_allocations]
        )
        baseline_cost = cost_model.max_cost(num_layers)
        logger.info(
            "Cost range: %.4f – %.4f (baseline=%.4f)",
            float(alloc_costs.min()),
            float(alloc_costs.max()),
            baseline_cost,
        )

        band_results = compute_dp_per_band(
            surrogate, all_allocations, alloc_costs,
            num_layers, baseline_cost, cost_model,
            landscape.num_cost_bins,
        )
        band_edges = compute_band_edges(
            alloc_costs, baseline_cost, landscape.num_cost_bins,
        )
        logger.info(
            "DP solutions for %d / %d cost bands",
            len(band_results),
            landscape.num_cost_bins,
        )

        # 2. Per-band search with real evaluator (incremental save + resume)
        bands_dir = output_dir / "bands"

        from contextlib import nullcontext

        backend_cm = self.config.backend.create_backend() if self.config.backend else nullcontext()
        async with backend_cm as backend:
            async with self.config.evaluators.create_evaluators(
                output_dir, backend=backend
            ) as evals:
                validated_bands: list[BandValidationResult] = []
                total_evals = 0

                for step, band in enumerate(band_results):
                    band_dir = bands_dir / f"band_{band.band_index:03d}"
                    result_path = band_dir / "result.json"

                    # Skip if already completed (resumption)
                    if result_path.exists():
                        logger.info(
                            "Band %d already validated, skipping (step %d / %d)",
                            band.band_index,
                            step + 1,
                            len(band_results),
                        )
                        loaded_result = json.loads(result_path.read_text())
                        surrogate_dp = PlacementResult.model_validate_json(
                            (band_dir / "surrogate_dp.json").read_text()
                        )
                        # Support both old (best_score: float) and new (best_scores: dict) formats
                        raw_scores = loaded_result.get("best_scores", loaded_result.get("best_score"))
                        if isinstance(raw_scores, dict):
                            loaded_score = sum(raw_scores.values()) / len(raw_scores)
                        else:
                            loaded_score = float(raw_scores)
                        validated_bands.append(
                            BandValidationResult(
                                band_index=band.band_index,
                                cost_low=band.cost_low,
                                cost_high=band.cost_high,
                                surrogate_dp=surrogate_dp,
                                validated=PlacementResult(
                                    cost=loaded_result["final_cost"],
                                    score=loaded_score,
                                    assignments=tuple(
                                        loaded_result["best_parameters"]["assignments"]
                                    ),
                                ),
                                total_evaluations=loaded_result["total_evaluations"],
                            )
                        )
                        total_evals += loaded_result["total_evaluations"]
                        continue

                    logger.info(
                        "Validating band %d / %d (cost %.4f – %.4f, surrogate score %.4f)",
                        step + 1,
                        len(band_results),
                        band.cost_low,
                        band.cost_high,
                        band.score,
                    )

                    # Save surrogate DP first (before the search that might crash)
                    band_dir.mkdir(parents=True, exist_ok=True)
                    surrogate_dp = PlacementResult(
                        cost=band.cost,
                        score=band.score,
                        assignments=band.assignments,
                    )
                    (band_dir / "surrogate_dp.json").write_text(
                        surrogate_dp.model_dump_json(indent=2)
                    )

                    # Construct per-band problem
                    problem_config = MaxPerformanceProblemConfig(
                        problem_type="cost_max_performance",
                        cost_model=landscape.cost_model,
                        cost_budget=band.cost_high,
                        cost_floor=band.cost_low,
                    )
                    problem = CostConstrainedProblem(
                        evaluators=evals, config=problem_config
                    )

                    # Construct per-band search strategy
                    search_config = ViterbiSurrogateSearchConfig(
                        strategy_type="viterbi_surrogate",
                        exploration_dataset_path=landscape.exploration_dataset_path,
                        num_rounds=self.config.search_num_rounds,
                        evals_per_round=self.config.search_evals_per_round,
                        candidates_per_model=self.config.search_candidates_per_model,
                        cv_folds=self.config.search_cv_folds,
                        ei_xi=self.config.search_ei_xi,
                        seed=self.config.search_seed,
                    )
                    strategy = search_config.create_search_strategy()
                    result = await strategy.search(problem)

                    # Save per-band result (CostConstrainedSearchResult as JSON)
                    result_path.write_text(result.model_dump_json(indent=2))

                    total_evals += result.total_evaluations

                    # Aggregate dict scores to scalar for PlacementResult
                    validated_score = sum(result.best_scores.values()) / len(result.best_scores)

                    validated_bands.append(
                        BandValidationResult(
                            band_index=band.band_index,
                            cost_low=band.cost_low,
                            cost_high=band.cost_high,
                            surrogate_dp=surrogate_dp,
                            validated=PlacementResult(
                                cost=result.final_cost,
                                score=validated_score,
                                assignments=result.best_parameters.assignments,
                            ),
                            total_evaluations=result.total_evaluations,
                        )
                    )

                    logger.info(
                        "Band %d: surrogate=%.4f, validated=%.4f, evals=%d",
                        band.band_index,
                        band.score,
                        validated_score,
                        result.total_evaluations,
                    )

        # 3. Compute summary statistics
        if len(validated_bands) >= 2:
            surrogate_scores = [b.surrogate_dp.score for b in validated_bands]
            real_scores = [b.validated.score for b in validated_bands]
            corr_result = spearmanr(surrogate_scores, real_scores)
            spearman_rho = float(corr_result.statistic)
            if np.isnan(spearman_rho):
                spearman_rho = 0.0
            residuals = [r - s for s, r in zip(surrogate_scores, real_scores)]
            mean_residual = float(np.mean(residuals))
        else:
            spearman_rho = 0.0
            mean_residual = 0.0

        summary = ValidationSummary(
            spearman_rho=spearman_rho,
            mean_residual=mean_residual,
            total_bands=len(validated_bands),
            total_evaluations=total_evals,
        )

        # 4. Assemble and save final validation_results.json
        results = ValidationResults(
            num_layers=num_layers,
            baseline_cost=baseline_cost,
            num_cost_bins=landscape.num_cost_bins,
            cost_bin_edges=band_edges.tolist(),
            bands=validated_bands,
            summary=summary,
        )
        results_path = output_dir / "validation_results.json"
        results.save(results_path)
        logger.info(
            "Saved validation results: %d bands, rho=%.4f, total_evals=%d -> %s",
            len(validated_bands),
            spearman_rho,
            total_evals,
            results_path,
        )


# ============================================================================
# Config
# ============================================================================


class ValidateLandscapeCommandConfig(BaseCommandConfig):
    """Configuration for validate_landscape command.

    Runs per-band ViterbiSurrogateSearch with real evaluation to validate
    surrogate predictions from placement_landscape.
    """

    _impl_class: ClassVar[type[BaseCommand]] = ValidateLandscapeCommand

    command: Literal["validate_landscape"] = "validate_landscape"

    backend: BackendConfig | None = Field(
        default=None,
        description="Backend configuration. None for mock evaluators.",
    )

    landscape: LandscapeConfig = Field(
        description="Shared landscape configuration (binning, surrogate, cost model).",
    )
    evaluators: EvaluatorsConfig = Field(description="Evaluator configuration.")

    # Per-band search (ViterbiSurrogateSearch params, inlined)
    search_num_rounds: int = Field(
        description="Surrogate refinement rounds per band.",
        gt=0,
    )
    search_evals_per_round: int = Field(
        description="Real evaluations per round per band.",
        gt=0,
    )
    search_candidates_per_model: int = Field(
        description="Viterbi top-N candidates per bootstrap model.",
        gt=0,
    )
    search_cv_folds: int = Field(
        description="Cross-validation folds for surrogate quality.",
        gt=1,
    )
    search_ei_xi: float = Field(
        description="Exploration parameter for Expected Improvement.",
        ge=0,
    )
    search_seed: int = Field(
        description="Random seed for per-band search.",
    )
