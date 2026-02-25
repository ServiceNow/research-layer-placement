"""Optimize command for layer placement."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import ClassVar, Literal

from hydra.core.hydra_config import HydraConfig
from pydantic import Field

from place_layers.backend import BackendConfig
from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.evaluator.evaluators import EvaluatorsConfig
from place_layers.problem import ProblemConfig
from place_layers.result_handler import ResultHandlerConfig
from place_layers.search_strategy import SearchStrategyConfig
from place_layers.search_strategy.progress import ObserversConfig


logger = logging.getLogger(__name__)


class OptimizeCommand(BaseCommand):
    """Run layer placement optimization."""

    config: "OptimizeCommandConfig"

    async def run(self) -> None:
        """Execute optimization."""
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)

        logger.info(f"Output directory: {output_dir}")
        logger.info("Starting optimization run...")

        backend_cm = self.config.backend.create_backend() if self.config.backend else nullcontext()
        async with backend_cm as backend:
            async with self.config.evaluators.create_evaluators(
                output_dir, backend=backend
            ) as evals:
                result = None
                for key in evals.objective_names:
                    problem = self.config.problem.create_problem(evals, objective_name=key)

                    # Pre-populate eval log with supplemental data for snowball.
                    # Strategy reads pre-seeded records on startup.
                    eval_log = output_dir / f"evaluations_{key}.jsonl"
                    n_prepopulated = 0
                    if self.config.supplemental_data_dirs:
                        from place_layers.surrogate.evaluation_log import (
                            append_record,
                            load_pooled_data,
                            resolve_supplemental_dirs,
                        )

                        supp_files = resolve_supplemental_dirs(
                            self.config.supplemental_data_dirs, key
                        )
                        if supp_files:
                            pool = load_pooled_data(supp_files, deduplicate=True)
                            eval_log.parent.mkdir(parents=True, exist_ok=True)
                            with eval_log.open("w") as f:
                                for rec in pool.as_records():
                                    append_record(f, rec)
                            n_prepopulated = pool.num_samples
                            logger.info(
                                "Pre-populated eval log with %d supplemental records -> %s",
                                n_prepopulated, eval_log,
                            )

                    # Strategy only needs evaluation_log_path — it reads pre-seeded records on startup
                    search_cfg = self.config.search_strategy.model_copy(
                        update={"evaluation_log_path": str(eval_log)}
                    )

                    # Create progress estimator
                    progress_estimator = search_cfg.create_progress_estimator(
                        problem
                    )

                    with self.config.observers.create_observers() as observers:
                        search_strategy = search_cfg.create_search_strategy(
                            observers=observers,
                            progress_estimator=progress_estimator,
                        )
                        result = await search_strategy.search(
                            problem=problem, dry_run=self.config.dry_run
                        )

                    # Update eval log sidecar to reflect pre-populated + strategy records
                    if n_prepopulated > 0 and result is not None:
                        from place_layers.surrogate.evaluation_log import (
                            load_meta,
                            write_meta,
                        )

                        meta = load_meta(eval_log)
                        if meta is not None:
                            write_meta(
                                eval_log,
                                meta.model_copy(update={
                                    "num_records": (meta.num_records or 0) + n_prepopulated,
                                }),
                            )

                if result is None:
                    return

                # Log per-evaluator score breakdown for multi-evaluator runs
                if len(evals.evaluators) > 1:
                    raw_scores = await evals.evaluate(result.best_parameters)
                    logger.info("Best result breakdown:")
                    for name, score in raw_scores.items():
                        logger.info(f"  {name}: {score:.6f}")

                result_handler = self.config.result_handler.create_result_handler()
                output_path = output_dir / "results.json"
                await result_handler.save(result=result, output_path=output_path)

        logger.info("Optimization complete!")


class OptimizeCommandConfig(BaseCommandConfig):
    """Configuration for optimize command."""

    _impl_class: ClassVar[type[BaseCommand]] = OptimizeCommand

    command: Literal["optimize"] = "optimize"

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
            "Directories containing pre-existing evaluation data to pre-populate "
            "the eval log before search begins. Each directory is searched "
            "recursively for evaluations_{obj}.jsonl snowball files."
        ),
    )

    dry_run: bool = Field(
        description="Show estimated eval counts without running optimization",
    )
