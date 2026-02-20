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

                    # Create progress estimator
                    progress_estimator = self.config.search_strategy.create_progress_estimator(
                        problem
                    )

                    with self.config.observers.create_observers() as observers:
                        search_strategy = self.config.search_strategy.create_search_strategy(
                            observers=observers,
                            progress_estimator=progress_estimator,
                        )
                        result = await search_strategy.search(
                            problem=problem, dry_run=self.config.dry_run
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

    dry_run: bool = Field(
        description="Show estimated eval counts without running optimization",
    )
