"""Optimize command for layer placement."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import ClassVar, Literal

from hydra.core.hydra_config import HydraConfig
from pydantic import Field

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.evaluator.multi_evaluator_config import EvaluatorsConfig
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
        # Get output directory from Hydra's runtime config
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)

        logger.info(f"Output directory: {output_dir}")
        logger.info("Starting optimization run...")

        # Create evaluators (can be single or multiple with aggregation)
        async with self.config.evaluators.create_evaluators(
            output_dir
        ) as evaluator_instances:
            # Create component instances using smart constructors
            # Architecture: Problem wraps Evaluator, Search only sees Problem
            # For multi-evaluator: evaluator_instances.evaluate() uses aggregation
            problem = self.config.problem.create_problem(evaluator_instances)

            # Create progress estimator (returns None if not supported by strategy/problem)
            progress_estimator = self.config.search_strategy.create_progress_estimator(
                problem
            )

            # Create observers from config (context manager handles lifecycle)
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

            # Optionally log individual evaluator scores for best result (in parallel)
            if len(evaluator_instances.evaluators) > 1 and result.best_placement is not None:
                logger.info("Best result breakdown:")
                evaluator_tasks = [
                    evaluator.evaluate(result.best_placement)
                    for evaluator in evaluator_instances.evaluators.values()
                ]
                scores = await asyncio.gather(*evaluator_tasks)

                for name, score in zip(evaluator_instances.evaluators.keys(), scores):
                    logger.info(f"  {name}: {score:.6f}")

            result_handler = self.config.result_handler.create_result_handler()
            output_path = output_dir / "results.json"
            await result_handler.save(result=result, output_path=output_path)

        logger.info("Optimization complete!")


class OptimizeCommandConfig(BaseCommandConfig):
    """Configuration for optimize command."""

    _impl_class: ClassVar[type[BaseCommand]] = OptimizeCommand

    command: Literal["optimize"] = "optimize"

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
