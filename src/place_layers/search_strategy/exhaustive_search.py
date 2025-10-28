"""Exhaustive search strategy implementation."""

import logging
from itertools import combinations
from typing import ClassVar, Literal

from pydantic import Field

from place_layers.core.types import SearchResult
from place_layers.problem.base_problem import BaseProblem
from place_layers.problem.layer_placement import (
    LayerPlacement,
    LayerPlacementProblemConfig,
)
from place_layers.search_strategy.base_search_strategy import BaseSearchStrategy
from place_layers.search_strategy.base_search_strategy_config import (
    BaseSearchStrategyConfig,
)

logger = logging.getLogger(__name__)


class ExhaustiveSearch(BaseSearchStrategy[LayerPlacement]):
    """
    Exhaustive search over all valid configurations.

    Enumerates and evaluates every configuration that satisfies budget constraints,
    returning the globally optimal solution. This provides:
    - Ground truth for validating heuristic search algorithms
    - Landscape analysis (score distributions, local optima count)
    - Optimality gap measurements

    WARNING: Only feasible for small problems. Search space grows combinatorially:
    - 10 layers, budgets=[2,3]: ~2,500 configs
    - 20 layers, budgets=[5,8]: ~100M configs (borderline)
    - 50 layers, budgets=[4,8]: ~10^15 configs (IMPOSSIBLE)
    """

    def __init__(self, config: "ExhaustiveSearchConfig") -> None:
        super().__init__(config)
        self.config: ExhaustiveSearchConfig  # Type narrowing

    async def search(
        self,
        problem: BaseProblem[LayerPlacement],
    ) -> SearchResult[LayerPlacement]:
        """
        Run exhaustive search.

        Args:
            problem: Problem definition with embedded evaluator and budget constraints.

        Returns:
            Search results with globally optimal configuration.

        Raises:
            ValueError: If search space exceeds max_configs limit.
        """
        # Extract problem parameters
        if not isinstance(problem.config, LayerPlacementProblemConfig):
            raise TypeError(
                f"ExhaustiveSearch requires LayerPlacementProblemConfig, "
                f"got {type(problem.config)}"
            )

        # Get dimensions from evaluator
        num_layers = problem.evaluator.config.num_layers
        num_mixers = problem.evaluator.config.num_mixers
        budgets = problem.config.budgets

        logger.info(f"\n{'='*60}")
        logger.info("Exhaustive Search Configuration")
        logger.info(f"{'='*60}")
        logger.info(f"Total layers: {num_layers}")
        logger.info(f"Number of mixer types: {num_mixers}")
        logger.info(f"Budgets: {budgets}")

        # Estimate search space size
        search_space_size = self._estimate_search_space(num_layers, budgets)
        logger.info(f"Estimated configurations: {search_space_size:,}")

        if search_space_size > self.config.max_configs:
            raise ValueError(
                f"Search space too large: {search_space_size:,} configurations "
                f"exceeds limit of {self.config.max_configs:,}. "
                f"Exhaustive search only works for small problems."
            )

        # Generate all valid configurations
        logger.info("\nEnumerating all valid configurations...")
        all_configs = list(
            self._generate_all_valid_configs(num_layers, num_mixers, budgets)
        )
        logger.info(f"Generated {len(all_configs):,} configurations")

        # Evaluate all configurations
        logger.info("\nEvaluating all configurations...")
        scores = await problem.evaluate_batch(all_configs)

        # Find best configuration
        if self.config.higher_is_better:
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
        else:
            best_idx = min(range(len(scores)), key=lambda i: scores[i])

        best_config = all_configs[best_idx]
        best_score = scores[best_idx]

        logger.info(f"\n{'='*60}")
        logger.info("EXHAUSTIVE SEARCH COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Evaluated: {len(all_configs):,} configurations")
        logger.info(f"Best score: {best_score:.4f}")
        logger.info(f"Best configuration: {best_config.assignments}")

        # Optionally log landscape statistics
        if self.config.log_statistics:
            self._log_landscape_statistics(scores, best_score)

        return SearchResult(
            best_parameters=best_config,
            best_score=best_score,
            total_evaluations=len(all_configs),
        )

    def _estimate_search_space(
        self, num_layers: int, budgets: list[int]
    ) -> int:
        """
        Estimate number of valid configurations.

        For budgets=[b0, b1, ..., bk]:
        Size = C(n, b0) × C(n-b0, b1) × ... × C(n-b0-b1-..., bk)
        """
        from math import comb

        size = 1
        remaining = num_layers

        for budget in budgets:
            size *= comb(remaining, budget)
            remaining -= budget

        return size

    def _generate_all_valid_configs(
        self, num_layers: int, num_mixers: int, budgets: list[int]
    ) -> list[LayerPlacement]:
        """
        Generate all layer placements satisfying budget constraints.

        Uses recursive enumeration:
        1. Choose budgets[0] layers for mixer[0]
        2. From remaining, choose budgets[1] layers for mixer[1]
        3. Continue for all budget phases
        4. Assign remaining layers to last mixer
        """
        configs = []

        def enumerate_recursive(
            current_assignment: list[int],
            available_layers: list[int],
            phase_idx: int,
        ) -> None:
            """Recursively enumerate layer placements."""
            if phase_idx >= len(budgets):
                # Base case: assign remaining layers to last mixer
                final_assignment = current_assignment.copy()
                for layer in available_layers:
                    final_assignment[layer] = num_mixers - 1

                configs.append(
                    LayerPlacement(assignments=tuple(final_assignment))
                )
                return

            # Choose budgets[phase_idx] layers for mixer[phase_idx]
            budget = budgets[phase_idx]
            mixer_idx = phase_idx

            for selected_layers in combinations(available_layers, budget):
                # Create new assignment with selected layers
                new_assignment = current_assignment.copy()
                for layer in selected_layers:
                    new_assignment[layer] = mixer_idx

                # Recurse with remaining layers
                new_available = [
                    l for l in available_layers if l not in selected_layers
                ]
                enumerate_recursive(new_assignment, new_available, phase_idx + 1)

        # Initialize with all layers unassigned
        initial_assignment = [-1] * num_layers
        initial_available = list(range(num_layers))

        enumerate_recursive(initial_assignment, initial_available, 0)

        return configs

    def _log_landscape_statistics(
        self, scores: list[float], best_score: float
    ) -> None:
        """Log statistics about the score landscape."""
        import statistics

        logger.info("\nLandscape Statistics:")
        logger.info(f"  Min score: {min(scores):.4f}")
        logger.info(f"  Max score: {max(scores):.4f}")
        logger.info(f"  Mean score: {statistics.mean(scores):.4f}")
        logger.info(f"  Std dev: {statistics.stdev(scores):.4f}")

        # Count near-optimal configurations
        threshold = 0.95 if self.config.higher_is_better else 1.05
        if self.config.higher_is_better:
            near_optimal = sum(s >= threshold * best_score for s in scores)
        else:
            near_optimal = sum(s <= threshold * best_score for s in scores)

        logger.info(
            f"  Configs within 5% of optimal: {near_optimal} "
            f"({near_optimal/len(scores)*100:.1f}%)"
        )


class ExhaustiveSearchConfig(BaseSearchStrategyConfig):
    """Configuration for exhaustive search."""

    _impl_class: ClassVar[type[BaseSearchStrategy]] = ExhaustiveSearch

    strategy_type: Literal["exhaustive"] = "exhaustive"

    max_configs: int = Field(
        default=100_000,
        description="Maximum number of configurations to evaluate. "
        "Safety limit to prevent accidentally running exhaustive search "
        "on problems that are too large.",
        gt=0,
    )

    higher_is_better: bool = Field(
        default=True,
        description="Whether higher scores are better",
    )

    log_statistics: bool = Field(
        default=True,
        description="Whether to log landscape statistics (min, max, mean, std)",
    )
