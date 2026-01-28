"""Optimize command for layer placement."""

import logging
from pathlib import Path
from typing import ClassVar, Literal

from hydra.core.hydra_config import HydraConfig
from pydantic import Field

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.evaluator import EvaluatorConfig
from place_layers.problem import ProblemConfig
from place_layers.result_handler import ResultHandlerConfig
from place_layers.search_strategy import SearchStrategyConfig


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

        # Create component instances using smart constructors
        # Architecture: Problem wraps Evaluator, Search only sees Problem
        async with self.config.evaluator.create_evaluator(output_dir) as evaluator:
            problem = self.config.problem.create_problem(evaluator)
            search_strategy = self.config.search_strategy.create_search_strategy()
            result_handler = self.config.result_handler.create_result_handler()

            # Run the search (search only interacts with problem, not evaluator directly)
            result = await search_strategy.search(problem=problem)

            # Save results
            output_path = output_dir / "results.json"
            await result_handler.save(result=result, output_path=output_path)

        logger.info("Optimization complete!")


class OptimizeCommandConfig(BaseCommandConfig):
    """Configuration for optimize command."""

    _command_class: ClassVar[type[BaseCommand]] = OptimizeCommand

    command: Literal["optimize"] = "optimize"

    problem: ProblemConfig = Field(description="Problem configuration")

    evaluator: EvaluatorConfig = Field(description="Evaluator configuration")

    search_strategy: SearchStrategyConfig = Field(
        description="Search strategy configuration"
    )

    result_handler: ResultHandlerConfig = Field(
        description="Result handler configuration"
    )
