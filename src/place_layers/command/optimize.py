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


if TYPE_CHECKING:
    from place_layers.problem.base_problem import BaseProblem
    from place_layers.search_strategy.progress import SearchProgressEstimator


logger = logging.getLogger(__name__)


class OptimizeCommand(BaseCommand):
    """Run layer placement optimization."""

    config: "OptimizeCommandConfig"

    async def run(self) -> None:
        """Execute optimization."""
        from place_layers.search_strategy.progress import RichSearchDisplay

        # Get output directory from Hydra's runtime config
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)

        logger.info(f"Output directory: {output_dir}")

        # Create component instances using smart constructors
        # Architecture: Problem wraps Evaluator, Search only sees Problem
        async with self.config.evaluator.create_evaluator(output_dir) as evaluator:
            problem = self.config.problem.create_problem(evaluator)

            # Create progress estimator (returns None if not supported by strategy/problem)
            progress_estimator = self.config.search_strategy.create_progress_estimator(
                problem
            )

            if self.config.dry_run:
                # Display estimates and exit without running optimization
                self._display_dry_run_info(problem, progress_estimator)
                return

            logger.info("Starting optimization run...")
            result_handler = self.config.result_handler.create_result_handler()

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

    def _display_dry_run_info(
        self,
        problem: "BaseProblem",
        progress_estimator: "SearchProgressEstimator | None",
    ) -> None:
        """Display estimated eval counts without running optimization."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        from place_layers.problem.layer_placement import (
            CostConstrainedProblem,
            MaxPerformanceProblemConfig,
            MinCostProblemConfig,
        )

        console = Console()

        # Build problem info table
        info_table = Table.grid(padding=(0, 2))
        info_table.add_column(style="bold cyan")
        info_table.add_column()

        # Problem type and constraint
        if isinstance(problem, CostConstrainedProblem):
            if isinstance(problem.config, MaxPerformanceProblemConfig):
                info_table.add_row("Mode", "MAX PERFORMANCE")
                info_table.add_row("Constraint", f"Cost ≤ {problem.config.cost_budget:.4f}")
            elif isinstance(problem.config, MinCostProblemConfig):
                info_table.add_row("Mode", "MIN COST")
                info_table.add_row(
                    "Constraint", f"Performance ≥ {problem.config.performance_target:.4f}"
                )
            # Cost bounds
            min_cost = problem.cost_model.min_cost(problem.evaluator.num_layers)
            max_cost = problem.cost_model.max_cost(problem.evaluator.num_layers)
            info_table.add_row("Cost bounds", f"[{min_cost:.4f}, {max_cost:.4f}]")

        info_table.add_row("Layers", str(problem.evaluator.num_layers))
        info_table.add_row("Mixers", ", ".join(problem.evaluator.mixer_names))

        # Eval count estimates
        if progress_estimator is not None:
            initial = progress_estimator.initial
            info_table.add_row(
                "Expected evals",
                f"~{initial.expected:,}",
            )
            info_table.add_row(
                "95% CI",
                f"[{initial.lower:,}, {initial.upper:,}]",
            )
            info_table.add_row(
                "Allocations",
                str(len(progress_estimator.allocation_info)),
            )
        else:
            info_table.add_row("Eval estimates", "[dim]Not available for this strategy[/dim]")

        # Print main panel
        console.print()
        console.print(
            Panel(
                info_table,
                title="[bold]Dry Run - Optimization Estimates[/bold]",
                border_style="blue",
            )
        )

        # Per-allocation breakdown (if available and not too many)
        if progress_estimator is not None and len(progress_estimator.allocation_info) <= 20:
            alloc_table = Table(
                show_header=True,
                header_style="bold",
                padding=(0, 1),
            )
            alloc_table.add_column("#", justify="right", style="dim")
            alloc_table.add_column("Allocation")
            alloc_table.add_column("Expected", justify="right")
            alloc_table.add_column("95% CI", justify="right")

            mixer_names = problem.evaluator.mixer_names
            for i, info in enumerate(progress_estimator.allocation_info, 1):
                alloc_dict = dict(zip(mixer_names, info.allocation, strict=True))
                alloc_str = ", ".join(f"{k}={v}" for k, v in alloc_dict.items())
                alloc_table.add_row(
                    str(i),
                    f"{{{alloc_str}}}",
                    f"~{int(info.expected):,}",
                    f"[{info.estimated_lower:,}, {info.estimated_upper:,}]",
                )

            console.print()
            console.print(
                Panel(
                    alloc_table,
                    title="[bold]Per-Allocation Breakdown[/bold]",
                    border_style="yellow",
                )
            )

        console.print()
        console.print(
            Text("Dry run complete. Use dry_run=false to run optimization.", style="dim")
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

    dry_run: bool = Field(
        description="Show estimated eval counts without running optimization",
    )
