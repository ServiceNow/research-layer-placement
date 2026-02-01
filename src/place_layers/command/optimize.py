"""Optimize command for layer placement."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from hydra.core.hydra_config import HydraConfig
from pydantic import Field

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.evaluator import EvaluatorConfig
from place_layers.problem import ProblemConfig
from place_layers.result_handler import ResultHandlerConfig
from place_layers.search_strategy import SearchStrategyConfig
from place_layers.search_strategy.progress.estimator_config import (
    ProgressEstimatorConfig,
)

if TYPE_CHECKING:
    from place_layers.problem.layer_placement import CostConstrainedProblem
    from place_layers.search_strategy.progress.estimator import SearchProgressEstimator


logger = logging.getLogger(__name__)


class OptimizeCommand(BaseCommand):
    """Run layer placement optimization."""

    config: "OptimizeCommandConfig"

    async def run(self) -> None:
        """Execute optimization."""
        from place_layers.problem.layer_placement import CostConstrainedProblem
        from place_layers.search_strategy.progress import RichSearchDisplay

        # Get output directory from Hydra's runtime config
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)

        logger.info(f"Output directory: {output_dir}")
        logger.info("Starting optimization run...")

        # Create component instances using smart constructors
        # Architecture: Problem wraps Evaluator, Search only sees Problem
        async with self.config.evaluator.create_evaluator(output_dir) as evaluator:
            problem = self.config.problem.create_problem(evaluator)
            result_handler = self.config.result_handler.create_result_handler()

            # Create progress estimator if configured and problem is cost-constrained
            progress_estimator = None
            if (
                self.config.progress_estimator is not None
                and isinstance(problem, CostConstrainedProblem)
            ):
                progress_estimator = self._create_progress_estimator(problem)

            # Use Rich display for progress visualization
            with RichSearchDisplay() as display:
                search_strategy = self.config.search_strategy.create_search_strategy(
                    observer=display,
                    progress_estimator=progress_estimator,
                )

                # Run the search (search only interacts with problem, not evaluator directly)
                result = await search_strategy.search(problem=problem)

            # Save results
            output_path = output_dir / "results.json"
            await result_handler.save(result=result, output_path=output_path)

        logger.info("Optimization complete!")

    def _create_progress_estimator(
        self, problem: "CostConstrainedProblem"
    ) -> "SearchProgressEstimator | None":
        """Create progress estimator from config.

        Args:
            problem: The cost-constrained optimization problem.

        Returns:
            Configured SearchProgressEstimator, or None if not applicable.
        """
        from place_layers.problem.layer_placement import MaxPerformanceProblemConfig

        if self.config.progress_estimator is None:
            return None

        config = problem.config
        cost_model = problem.cost_model
        num_layers = problem.evaluator.num_layers
        mixer_names = problem.evaluator.mixer_names
        phase_order = cost_model.phase_order

        # Only support max_performance mode for now (min_cost has early termination)
        if not isinstance(config, MaxPerformanceProblemConfig):
            return None

        # Build name-to-allocation-index mapping
        name_to_alloc_idx = {name: i for i, name in enumerate(mixer_names)}

        # Get beam parameters from search config
        search_config = self.config.search_strategy
        beam_width = getattr(search_config, "beam_width", 1)
        cross_phase_beam_width = getattr(search_config, "cross_phase_beam_width", 1)
        max_allocations = getattr(search_config, "max_allocations", 100)

        # Enumerate allocations in the same order as search will use
        allocations: list[tuple[int, ...]] = []
        for alloc_cost, allocation in cost_model.allocations_by_cost(
            num_layers, direction="decreasing"
        ):
            if alloc_cost > config.cost_budget:
                continue
            if len(allocations) >= max_allocations:
                break
            allocations.append(allocation)

        if not allocations:
            return None

        def phase_budgets_fn(allocation: tuple[int, ...]) -> list[int]:
            """Get phase budgets for an allocation."""
            return [
                allocation[name_to_alloc_idx[phase_order[i]]]
                for i in range(len(phase_order) - 1)
            ]

        return self.config.progress_estimator.create_estimator(
            num_layers=num_layers,
            allocations=allocations,
            phase_budgets_fn=phase_budgets_fn,
            beam_width=beam_width,
            cross_phase_beam_width=cross_phase_beam_width,
        )


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

    progress_estimator: ProgressEstimatorConfig | None = Field(
        default=None,
        description="Optional progress estimator configuration for progress display",
    )
