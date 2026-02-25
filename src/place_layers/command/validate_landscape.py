"""Validate landscape command: per-band real evaluation of surrogate predictions.

For each objective and cost band:
1. Compute surrogate DP optimal (MRF extraction, like placement_landscape)
2. Run Viterbi surrogate search restricted to that band (Ridge bootstrap+EI or Bayesian LCB + real eval)
3. Save both surrogate DP and validated search results to JSON

Per-band results are saved incrementally under ``bands_{obj_name}/band_NNN/`` so
that a crash mid-way loses only the current band.  On restart, completed bands
are detected and skipped automatically.

Input: explore_output_dir (contains exploration_results_{obj_name}.json per objective)
       + evaluator config + cost model config.
Output: validation_results_{obj_name}.json per objective in Hydra output dir.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, ClassVar, Literal

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
    objective_result_path,
)
from place_layers.cost_model.enumeration import enumerate_feasible_allocations
from place_layers.backend import BackendConfig
from place_layers.evaluator.evaluators import Evaluators, EvaluatorsConfig
from place_layers.problem.layer_placement import (
    CostConstrainedProblem,
    MaxPerformanceProblemConfig,
)
from place_layers.search_strategy.cost_constrained.viterbi_surrogate import (
    ViterbiBayesianSurrogateSearchConfig,
    ViterbiSurrogateSearchConfig,
)

logger = logging.getLogger(__name__)

ViterbiSearchConfig = Annotated[
    ViterbiSurrogateSearchConfig | ViterbiBayesianSurrogateSearchConfig,
    Field(discriminator="strategy_type"),
]


# ============================================================================
# Command
# ============================================================================


class ValidateLandscapeCommand(BaseCommand):
    """Validate placement landscape predictions against real evaluator."""

    config: "ValidateLandscapeCommandConfig"

    async def run(self) -> None:
        """Execute per-objective, per-band validation with incremental saving."""
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)
        logger.info("Output directory: %s", output_dir)

        landscape = self.config.landscape
        explore_dir = Path(landscape.explore_output_dir)

        # 1. Objective-independent: cost model + allocation enumeration
        cost_model = landscape.cost_model.create_cost_model()
        # We need num_layers from the first exploration dataset
        first_obj_name = self.config.evaluators.objective_names[0]
        first_dataset_path = objective_result_path(
            explore_dir, "exploration_results", first_obj_name,
        )
        from place_layers.surrogate.dataset import ExplorationDataset
        first_dataset = ExplorationDataset.load(first_dataset_path)
        num_layers = first_dataset.num_layers

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

        band_edges = compute_band_edges(
            alloc_costs, baseline_cost, landscape.num_cost_bins,
        )

        # 2. Per-objective validation (each objective gets its own surrogate)
        from contextlib import nullcontext

        backend_cm = self.config.backend.create_backend() if self.config.backend else nullcontext()
        async with backend_cm as backend:
            async with self.config.evaluators.create_evaluators(
                output_dir, backend=backend
            ) as evals:
                for obj_name in evals.objective_names:
                    await self._validate_objective(
                        evals=evals,
                        obj_name=obj_name,
                        explore_dir=explore_dir,
                        output_dir=output_dir,
                        all_allocations=all_allocations,
                        alloc_costs=alloc_costs,
                        num_layers=num_layers,
                        baseline_cost=baseline_cost,
                        band_edges=band_edges,
                    )

    async def _validate_objective(
        self,
        *,
        evals: Evaluators,
        obj_name: str,
        explore_dir: Path,
        output_dir: Path,
        all_allocations: list,
        alloc_costs: np.ndarray,
        num_layers: int,
        baseline_cost: float,
        band_edges: np.ndarray,
    ) -> None:
        """Validate a single objective: fit surrogate, per-band search, save."""
        from datetime import datetime, timezone

        from place_layers.surrogate.evaluation_log import (
            EvaluationLogMeta,
            EvaluationRecord,
            append_record,
            load_pooled_data,
            load_records,
            resolve_supplemental_dirs,
            write_meta,
        )

        landscape = self.config.landscape

        logger.info("=== Validating objective: %s ===", obj_name)

        # Resolve supplemental dirs for this objective
        supplemental_dirs = list(landscape.supplemental_data_dirs)

        # Load per-objective exploration dataset and fit surrogate
        dataset_path = objective_result_path(
            explore_dir, "exploration_results", obj_name,
        )
        _dataset, surrogate, cost_model = load_and_fit_surrogate(
            str(dataset_path),
            landscape.cost_model,
            landscape.surrogate,
            landscape.seed,
            supplemental_data_dirs=supplemental_dirs,
        )

        band_results = compute_dp_per_band(
            surrogate, all_allocations, alloc_costs,
            num_layers, baseline_cost, cost_model,
            landscape.num_cost_bins,
        )
        logger.info(
            "DP solutions for %d / %d cost bands (objective=%s)",
            len(band_results),
            landscape.num_cost_bins,
            obj_name,
        )

        # Per-band search with real evaluator (incremental save + resume)
        bands_dir = output_dir / f"bands_{obj_name}"
        validated_bands: list[BandValidationResult] = []
        total_evals = 0

        # Resolve external supplemental dirs to file-level paths
        supplemental_files = resolve_supplemental_dirs(supplemental_dirs, obj_name)

        # Load supplemental records once for pre-populating band eval logs
        supplemental_records: list[EvaluationRecord] = []
        if supplemental_files:
            pool = load_pooled_data(supplemental_files, deduplicate=True)
            supplemental_records = pool.as_records()

        # Track strategy-generated records across bands for cross-band accumulation
        # and merged snowball (avoids geometric duplication from re-reading pre-populated logs)
        strategy_records_all_bands: list[EvaluationRecord] = []

        # Open top-level snowball for incremental writes
        merged_jsonl = output_dir / f"evaluations_{obj_name}.jsonl"
        merged_meta = EvaluationLogMeta(
            version=1,
            objective_name=obj_name,
            num_layers=num_layers,
            mixer_names=list(cost_model.phase_order),
            source_command="validate_landscape",
            source_detail="incremental_bands_merged",
            created_at=datetime.now(timezone.utc).isoformat(),
            num_records=None,
        )
        write_meta(merged_jsonl, merged_meta)
        merged_record_count = 0
        merged_file = merged_jsonl.open("w")
        try:
            # Seed snowball with supplemental records
            for rec in supplemental_records:
                append_record(merged_file, rec)
                merged_record_count += 1

            for step, band in enumerate(band_results):
                band_dir = bands_dir / f"band_{band.band_index:03d}"
                result_path = band_dir / "result.json"
                eval_log_jsonl = band_dir / f"evaluations_{obj_name}.jsonl"

                # Skip if already completed (resumption)
                if result_path.exists():
                    logger.info(
                        "Band %d already validated, skipping (step %d / %d, objective=%s)",
                        band.band_index,
                        step + 1,
                        len(band_results),
                        obj_name,
                    )
                    loaded_result = json.loads(result_path.read_text())
                    surrogate_dp = PlacementResult.model_validate_json(
                        (band_dir / "surrogate_dp.json").read_text()
                    )
                    # Support both old (best_score: float) and new (best_scores: dict) formats
                    raw_scores = loaded_result.get("best_scores", loaded_result.get("best_score"))
                    if isinstance(raw_scores, dict):
                        loaded_score = next(iter(evals.objectives(raw_scores).values()))
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

                    # Extract strategy's own records from resumed band
                    if eval_log_jsonl.exists():
                        all_band_records = load_records(eval_log_jsonl)
                        n_strategy = loaded_result["total_evaluations"]
                        band_strategy_records = (
                            all_band_records[len(all_band_records) - n_strategy:]
                            if n_strategy > 0
                            else []
                        )
                        strategy_records_all_bands.extend(band_strategy_records)
                        for rec in band_strategy_records:
                            append_record(merged_file, rec)
                            merged_record_count += 1

                    continue

                logger.info(
                    "Validating band %d / %d (cost %.4f – %.4f, surrogate score %.4f, objective=%s)",
                    step + 1,
                    len(band_results),
                    band.cost_low,
                    band.cost_high,
                    band.score,
                    obj_name,
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
                    evaluators=evals, config=problem_config,
                    objective_name=obj_name,
                )

                # Pre-populate band eval log with supplemental + cross-band strategy records
                n_prepop = 0
                if supplemental_records or (self.config.accumulate_across_bands and strategy_records_all_bands):
                    eval_log_jsonl.parent.mkdir(parents=True, exist_ok=True)
                    with eval_log_jsonl.open("w") as f:
                        for rec in supplemental_records:
                            append_record(f, rec)
                            n_prepop += 1
                        if self.config.accumulate_across_bands:
                            for rec in strategy_records_all_bands:
                                append_record(f, rec)
                                n_prepop += 1

                # Construct per-band search strategy — reads pre-seeded eval log
                search_config = self.config.search_strategy.model_copy(
                    update={
                        "exploration_dataset_path": str(dataset_path),
                        "evaluation_log_path": str(eval_log_jsonl),
                    }
                )
                strategy = search_config.create_search_strategy()
                try:
                    result = await strategy.search(problem)
                except ValueError as exc:
                    logger.warning(
                        "Band %d: search found no feasible placement (%s), "
                        "skipping (objective=%s)",
                        band.band_index,
                        exc,
                        obj_name,
                    )
                    continue

                # Post-search cost guard: reject results outside band range
                if not (band.cost_low - 1e-9 <= result.final_cost <= band.cost_high + 1e-9):
                    logger.warning(
                        "Band %d: result cost %.4f outside band [%.4f, %.4f], "
                        "skipping (objective=%s)",
                        band.band_index,
                        result.final_cost,
                        band.cost_low,
                        band.cost_high,
                        obj_name,
                    )
                    continue

                # Save per-band result (CostConstrainedSearchResult as JSON)
                result_path.write_text(result.model_dump_json(indent=2))

                total_evals += result.total_evaluations

                validated_score = result.best_score

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
                    "Band %d: surrogate=%.4f, validated=%.4f, evals=%d (objective=%s)",
                    band.band_index,
                    band.score,
                    validated_score,
                    result.total_evaluations,
                    obj_name,
                )

                # Extract strategy's own records and track for cross-band + snowball
                if eval_log_jsonl.exists():
                    all_band_records = load_records(eval_log_jsonl)
                    band_strategy_records = all_band_records[n_prepop:]
                    strategy_records_all_bands.extend(band_strategy_records)
                    for rec in band_strategy_records:
                        append_record(merged_file, rec)
                        merged_record_count += 1
        finally:
            merged_file.close()
            try:
                write_meta(
                    merged_jsonl,
                    merged_meta.model_copy(update={"num_records": merged_record_count}),
                )
            except Exception:
                logger.warning("Failed to update snowball sidecar", exc_info=True)
            if merged_record_count > 0:
                logger.info(
                    "Wrote merged evaluation log: %d records -> %s",
                    merged_record_count,
                    merged_jsonl,
                )

        # Compute summary statistics
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

        # Assemble and save per-objective validation results
        results = ValidationResults(
            objective_name=obj_name,
            num_layers=num_layers,
            baseline_cost=baseline_cost,
            num_cost_bins=self.config.landscape.num_cost_bins,
            cost_bin_edges=band_edges.tolist(),
            bands=validated_bands,
            summary=summary,
        )
        results_path = objective_result_path(
            output_dir, "validation_results", obj_name,
        )
        results.save(results_path)
        logger.info(
            "Saved validation results for %s: %d bands, rho=%.4f, total_evals=%d -> %s",
            obj_name,
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

    search_strategy: ViterbiSearchConfig = Field(
        description="Per-band Viterbi search strategy configuration. "
        "exploration_dataset_path is overridden per-objective at runtime.",
    )

    accumulate_across_bands: bool = Field(
        description="If true, each band's search includes evaluation data from all prior bands "
        "in the same run. Introduces ordering dependence but provides neighborhood data.",
    )
