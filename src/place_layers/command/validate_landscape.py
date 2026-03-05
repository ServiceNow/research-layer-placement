"""Validate landscape command: per-band real evaluation via surrogate search.

For each objective and occupied cost band:
1. Run Viterbi surrogate search restricted to that band (Ridge bootstrap+EI or Bayesian LCB + real eval)
2. Save validated search results to JSON

Per-band results are saved incrementally under ``{obj_name}/bands/band_NNN/`` so
that a crash mid-way loses only the current band.  On restart, completed bands
are detected and skipped automatically.

Input: explore_output_dir (contains {obj_name}/exploration_results.json per objective)
       + evaluator config + cost model config.
Output: {obj_name}/validation_results.json per objective in Hydra output dir.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, ClassVar, Literal

import numpy as np
from hydra.core.hydra_config import HydraConfig
from pydantic import Field

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.search_strategy.cost_constrained.types import (
    CostConstrainedSearchResult,
)
from place_layers.landscape import (
    BandValidationResult,
    LandscapeConfig,
    PlacementResult,
    ValidationResults,
    ValidationSummary,
    compute_band_edges,
    compute_bin_assignments,
    find_objective_result,
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

from place_layers.surrogate.evaluation_log import (
    EVALUATION_RECORDS_FILENAME,
    EvaluationLogMeta,
    discover_evaluation_records,
    load_records,
    write_meta,
)

logger = logging.getLogger(__name__)


def _load_completed_band_records(
    bands_dir: Path,
    prior_band_indices: list[int],
) -> tuple[list[Path], int]:
    """Find eval log paths from completed prior bands.

    Returns (paths, total_records) where paths can be used as seed_data_paths.
    """
    paths: list[Path] = []
    total_records = 0
    for band_index in prior_band_indices:
        band_dir = bands_dir / f"band_{band_index:03d}"
        result_path = band_dir / "result.json"
        eval_log = band_dir / EVALUATION_RECORDS_FILENAME
        if not result_path.exists() or not eval_log.exists():
            continue
        paths.append(eval_log)
        total_records += len(load_records(eval_log))
    return paths, total_records


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
        first_dataset_path = find_objective_result(
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

        # Discover occupied cost bins via simple digitization
        bin_indices = compute_bin_assignments(
            alloc_costs, band_edges, landscape.num_cost_bins,
        )
        occupied_bins = sorted(set(int(b) for b in bin_indices))

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
                        cost_model=cost_model,
                        occupied_bins=occupied_bins,
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
        cost_model,
        occupied_bins: list[int],
        num_layers: int,
        baseline_cost: float,
        band_edges: np.ndarray,
    ) -> None:
        """Validate a single objective: per-band search, save."""
        landscape = self.config.landscape

        logger.info("=== Validating objective: %s ===", obj_name)

        # Exploration dataset path (needed for strategy's exploration_dataset_path)
        dataset_path = find_objective_result(
            explore_dir, "exploration_results", obj_name,
        )

        logger.info(
            "Occupied cost bins: %d / %d (objective=%s)",
            len(occupied_bins),
            landscape.num_cost_bins,
            obj_name,
        )

        # Resolve supplemental files via discovery (paths only, no loading)
        supplemental_files = (
            discover_evaluation_records(landscape.supplemental_data_dirs, obj_name)
            if landscape.supplemental_data_dirs else []
        )

        # Per-band search with real evaluator (incremental save + resume)
        bands_dir = output_dir / obj_name / "bands"
        validated_bands: list[BandValidationResult] = []
        total_evals = 0

        for step, band_index in enumerate(occupied_bins):
            cost_low = float(band_edges[band_index])
            cost_high = float(band_edges[band_index + 1])

            band_dir = bands_dir / f"band_{band_index:03d}"
            result_path = band_dir / "result.json"

            # Skip if already completed (resumption)
            if result_path.exists():
                logger.info(
                    "Band %d already validated, skipping (step %d / %d, objective=%s)",
                    band_index,
                    step + 1,
                    len(occupied_bins),
                    obj_name,
                )
                loaded = CostConstrainedSearchResult.model_validate_json(
                    result_path.read_text()
                )
                validated_bands.append(
                    BandValidationResult(
                        band_index=band_index,
                        cost_low=cost_low,
                        cost_high=cost_high,
                        validated=PlacementResult(
                            cost=loaded.final_cost,
                            score=loaded.best_score,
                            assignments=loaded.best_parameters.assignments,
                        ),
                        total_evaluations=loaded.total_evaluations,
                    )
                )
                total_evals += loaded.total_evaluations

                # Incremental save of validation results
                self._save_validation_results(
                    obj_name=obj_name,
                    validated_bands=validated_bands,
                    num_layers=num_layers,
                    baseline_cost=baseline_cost,
                    num_cost_bins=landscape.num_cost_bins,
                    band_edges=band_edges,
                    total_evals=total_evals,
                    output_dir=output_dir,
                )
                continue

            logger.info(
                "Validating band %d (step %d / %d, cost %.4f – %.4f, objective=%s)",
                band_index,
                step + 1,
                len(occupied_bins),
                cost_low,
                cost_high,
                obj_name,
            )

            band_dir.mkdir(parents=True, exist_ok=True)

            # Construct per-band problem
            problem_config = MaxPerformanceProblemConfig(
                problem_type="cost_max_performance",
                cost_model=landscape.cost_model,
                cost_budget=cost_high,
                cost_floor=cost_low,
            )
            problem = CostConstrainedProblem(
                evaluators=evals, config=problem_config,
                objective_name=obj_name,
            )

            # Build seed_data_paths: supplemental files + prior bands
            seed_paths: list[str] = []
            seed_paths.extend(str(f) for f in supplemental_files)

            if self.config.accumulate_across_bands:
                prior_indices = [
                    bi for bi in occupied_bins[:step]
                    if (bands_dir / f"band_{bi:03d}" / "result.json").exists()
                ]
                prior_eval_paths, _ = _load_completed_band_records(
                    bands_dir, prior_indices,
                )
                seed_paths.extend(str(p) for p in prior_eval_paths)

            eval_log_path = band_dir / EVALUATION_RECORDS_FILENAME

            # Write sidecar metadata (objective, band bounds, etc.)
            write_meta(eval_log_path, EvaluationLogMeta(
                version=1,
                objective_name=obj_name,
                num_layers=num_layers,
                mixer_names=list(cost_model.phase_order),
                source_command="validate_landscape",
                source_detail="per_band_search",
                created_at=datetime.now(timezone.utc).isoformat(),
                num_records=None,
                cost_floor=cost_low,
                cost_ceiling=cost_high,
                band_index=band_index,
            ))

            # Construct per-band search strategy with seed data
            search_config = self.config.search_strategy.model_copy(
                update={
                    "exploration_dataset_path": str(dataset_path),
                    "evaluation_log_path": str(eval_log_path),
                    "seed_data_paths": seed_paths,
                }
            )
            strategy = search_config.create_search_strategy()
            try:
                result = await strategy.search(problem)
            except ValueError as exc:
                logger.warning(
                    "Band %d: search found no feasible placement (%s), "
                    "skipping (objective=%s)",
                    band_index,
                    exc,
                    obj_name,
                )
                continue

            # Post-search cost guard: reject results outside band range
            if not (cost_low - 1e-9 <= result.final_cost <= cost_high + 1e-9):
                logger.warning(
                    "Band %d: result cost %.4f outside band [%.4f, %.4f], "
                    "skipping (objective=%s)",
                    band_index,
                    result.final_cost,
                    cost_low,
                    cost_high,
                    obj_name,
                )
                continue

            # Save per-band result (CostConstrainedSearchResult as JSON)
            result_path.write_text(result.model_dump_json(indent=2))

            total_evals += result.total_evaluations

            validated_score = result.best_score

            validated_bands.append(
                BandValidationResult(
                    band_index=band_index,
                    cost_low=cost_low,
                    cost_high=cost_high,
                    validated=PlacementResult(
                        cost=result.final_cost,
                        score=validated_score,
                        assignments=result.best_parameters.assignments,
                    ),
                    total_evaluations=result.total_evaluations,
                )
            )

            logger.info(
                "Band %d: validated=%.4f, evals=%d (objective=%s)",
                band_index,
                validated_score,
                result.total_evaluations,
                obj_name,
            )

            # Incremental save of validation results
            self._save_validation_results(
                obj_name=obj_name,
                validated_bands=validated_bands,
                num_layers=num_layers,
                baseline_cost=baseline_cost,
                num_cost_bins=landscape.num_cost_bins,
                band_edges=band_edges,
                total_evals=total_evals,
                output_dir=output_dir,
            )

        # Final save (covers edge case where no bands were validated)
        self._save_validation_results(
            obj_name=obj_name,
            validated_bands=validated_bands,
            num_layers=num_layers,
            baseline_cost=baseline_cost,
            num_cost_bins=landscape.num_cost_bins,
            band_edges=band_edges,
            total_evals=total_evals,
            output_dir=output_dir,
        )

    @staticmethod
    def _save_validation_results(
        *,
        obj_name: str,
        validated_bands: list[BandValidationResult],
        num_layers: int,
        baseline_cost: float,
        num_cost_bins: int,
        band_edges: np.ndarray,
        total_evals: int,
        output_dir: Path,
    ) -> None:
        """Compute summary and save validation_results (partial or final)."""
        summary = ValidationSummary(
            total_bands=len(validated_bands),
            total_evaluations=total_evals,
        )

        results = ValidationResults(
            objective_name=obj_name,
            num_layers=num_layers,
            baseline_cost=baseline_cost,
            num_cost_bins=num_cost_bins,
            cost_bin_edges=band_edges.tolist(),
            bands=validated_bands,
            summary=summary,
        )
        results_path = objective_result_path(
            output_dir, "validation_results", obj_name,
        )
        results.save(results_path)
        logger.info(
            "Saved validation results for %s: %d bands, total_evals=%d -> %s",
            obj_name,
            len(validated_bands),
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
