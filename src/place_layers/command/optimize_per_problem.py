"""Optimize each objective separately and save a comparison summary."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar, Literal

from hydra.core.hydra_config import HydraConfig
from pydantic import BaseModel, Field

from place_layers.backend import BackendConfig
from place_layers.command.base_command import BaseCommand, CommandError
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.core.trace_problem_split import build_per_problem_evaluators
from place_layers.evaluator.evaluators import EvaluatorsConfig
from place_layers.landscape.config import find_objective_result, objective_result_path
from place_layers.problem import ProblemConfig
from place_layers.result_handler import ResultHandlerConfig
from place_layers.search_strategy import SearchStrategyConfig
from place_layers.search_strategy.progress import ObserversConfig


logger = logging.getLogger(__name__)


class OptimizationSummaryItem(BaseModel):
    """Compact per-objective summary for multi-objective optimization runs."""

    objective_name: str
    problem_id: str
    result_path: str
    exploration_dataset_path: str
    best_score: float
    total_evaluations: int
    final_cost: float | None = None
    best_assignments: list[str]


class MultiObjectiveOptimizationSummary(BaseModel):
    """Root artifact for per-problem optimization runs."""

    items: list[OptimizationSummaryItem]


def resolve_exploration_dataset_path(
    exploration_dataset_path: str,
    objective_name: str,
    *,
    num_objectives: int,
) -> Path:
    """Resolve the exploration dataset path for a specific objective."""
    base_path = Path(exploration_dataset_path)
    if base_path.is_dir():
        return find_objective_result(base_path, "exploration_results", objective_name)

    if num_objectives > 1:
        raise CommandError(
            "optimize_per_problem requires "
            "search_strategy.exploration_dataset_path to point to the exploration "
            "output directory containing per-objective exploration_results.json files."
        )

    return base_path


class OptimizePerProblemCommand(BaseCommand):
    """Run optimization separately for each objective."""

    config: "OptimizePerProblemCommandConfig"

    async def run(self) -> None:
        """Execute per-objective optimization."""
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)

        logger.info(f"Output directory: {output_dir}")
        logger.info("Starting per-problem optimization run...")

        derived_evaluators, manifest = build_per_problem_evaluators(
            self.config.evaluators,
            output_dir / "_problem_traces",
        )
        (output_dir / "per_problem_manifest.json").write_text(
            manifest.model_dump_json(indent=2)
        )
        problem_id_by_objective = {
            item.objective_name: item.problem_id for item in manifest.items
        }

        backend_cm = self.config.backend.create_backend() if self.config.backend else nullcontext()
        async with backend_cm as backend:
            async with derived_evaluators.create_evaluators(
                output_dir, backend=backend
            ) as evals:
                num_objectives = len(evals.objective_names)
                result_handler = self.config.result_handler.create_result_handler()
                results: list[OptimizationSummaryItem] = []

                for key in evals.objective_names:
                    problem = self.config.problem.create_problem(
                        evals, objective_name=key
                    )

                    seed_paths: list[str] = []
                    if self.config.supplemental_data_dirs:
                        from place_layers.surrogate.evaluation_log import (
                            discover_evaluation_records,
                        )

                        supp_files = discover_evaluation_records(
                            self.config.supplemental_data_dirs,
                            key,
                            num_layers=evals.num_layers,
                            mixer_names=list(evals.mixer_names),
                        )
                        seed_paths.extend(str(f) for f in supp_files)
                        if supp_files:
                            logger.info(
                                "Discovered %d supplemental data files for objective '%s'",
                                len(supp_files),
                                key,
                            )

                    from place_layers.surrogate.evaluation_log import (
                        EVALUATION_RECORDS_FILENAME,
                        EvaluationLogMeta,
                        write_meta,
                    )

                    eval_log = output_dir / key / EVALUATION_RECORDS_FILENAME
                    eval_log.parent.mkdir(parents=True, exist_ok=True)

                    write_meta(
                        eval_log,
                        EvaluationLogMeta(
                            version=1,
                            objective_name=key,
                            num_layers=evals.num_layers,
                            mixer_names=list(evals.mixer_names),
                            source_command="optimize_per_problem",
                            source_detail="surrogate_search",
                            created_at=datetime.now(timezone.utc).isoformat(),
                            num_records=None,
                        ),
                    )

                    dataset_path = resolve_exploration_dataset_path(
                        self.config.search_strategy.exploration_dataset_path,
                        key,
                        num_objectives=num_objectives,
                    )
                    search_cfg = self.config.search_strategy.model_copy(
                        update={
                            "exploration_dataset_path": str(dataset_path),
                            "seed_data_paths": seed_paths,
                            "evaluation_log_path": str(eval_log),
                        }
                    )

                    progress_estimator = search_cfg.create_progress_estimator(problem)

                    with self.config.observers.create_observers() as observers:
                        search_strategy = search_cfg.create_search_strategy(
                            observers=observers,
                            progress_estimator=progress_estimator,
                        )
                        result = await search_strategy.search(
                            problem=problem,
                            dry_run=self.config.dry_run,
                        )

                    result_path = objective_result_path(output_dir, "results", key)
                    await result_handler.save(result=result, output_path=result_path)
                    results.append(
                        OptimizationSummaryItem(
                            objective_name=key,
                            problem_id=problem_id_by_objective[key],
                            result_path=str(result_path),
                            exploration_dataset_path=str(dataset_path),
                            best_score=result.best_score,
                            total_evaluations=result.total_evaluations,
                            final_cost=getattr(result, "final_cost", None),
                            best_assignments=list(result.best_parameters.assignments),
                        )
                    )

                if not results:
                    return

                await result_handler.save(
                    result=MultiObjectiveOptimizationSummary(items=results),
                    output_path=output_dir / "results.json",
                )

        logger.info("Per-problem optimization complete!")


class OptimizePerProblemCommandConfig(BaseCommandConfig):
    """Configuration for optimize_per_problem command."""

    _impl_class: ClassVar[type[BaseCommand]] = OptimizePerProblemCommand

    command: Literal["optimize_per_problem"] = "optimize_per_problem"

    backend: BackendConfig | None = Field(
        default=None,
        description="Backend configuration. None for mock evaluators.",
    )

    problem: ProblemConfig = Field(description="Problem configuration")

    evaluators: EvaluatorsConfig = Field(
        description="Evaluator configuration (single or multiple with aggregation)"
    )

    search_strategy: SearchStrategyConfig = Field(
        description="Search strategy configuration"
    )

    result_handler: ResultHandlerConfig = Field(
        description="Result handler configuration"
    )

    observers: ObserversConfig = Field(
        description="Search progress observers (display + logging)"
    )

    supplemental_data_dirs: list[str] = Field(
        description=(
            "Directories containing pre-existing evaluation data. "
            "Each directory is searched recursively for evaluation record files."
        ),
    )

    dry_run: bool = Field(
        description="Show estimated eval counts without running optimization",
    )
