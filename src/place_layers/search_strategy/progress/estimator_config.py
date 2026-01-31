"""Progress estimator configurations.

Each search algorithm that supports progress estimation has a matching config.
The config's smart constructor creates an estimator given problem parameters.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Annotated, Callable, Literal

from pydantic import BaseModel, Field

from place_layers.search_strategy.progress.estimator import (
    AllocationProgressInfo,
    ProgressEstimate,
    SearchProgressEstimator,
)


class BaseProgressEstimatorConfig(BaseModel):
    """Base config for progress estimators."""

    estimator_type: str

    @abstractmethod
    def create_estimator(
        self,
        num_layers: int,
        allocations: list[tuple[int, ...]],
        phase_budgets_fn: Callable[[tuple[int, ...]], list[int]],
        beam_width: int,
        cross_phase_beam_width: int,
    ) -> SearchProgressEstimator:
        """Create a progress estimator for the given problem.

        Args:
            num_layers: Number of layers in model.
            allocations: List of allocations that will be explored.
            phase_budgets_fn: Function to get phase budgets for an allocation.
            beam_width: Within-phase beam width.
            cross_phase_beam_width: Cross-phase beam width.

        Returns:
            Configured SearchProgressEstimator.
        """
        ...


class HierarchicalProgressEstimatorConfig(BaseProgressEstimatorConfig):
    """Progress estimator for hierarchical beam search."""

    estimator_type: Literal["hierarchical"] = "hierarchical"

    def create_estimator(
        self,
        num_layers: int,
        allocations: list[tuple[int, ...]],
        phase_budgets_fn: Callable[[tuple[int, ...]], list[int]],
        beam_width: int,
        cross_phase_beam_width: int,
    ) -> SearchProgressEstimator:
        """Create a progress estimator for hierarchical beam search.

        Uses the allocation_bounds formula from hierarchical_bounds.py to
        estimate evaluation counts for each allocation.

        Args:
            num_layers: Number of layers in model.
            allocations: List of allocations that will be explored.
            phase_budgets_fn: Function to get phase budgets for an allocation.
            beam_width: Within-phase beam width.
            cross_phase_beam_width: Cross-phase beam width.

        Returns:
            Configured SearchProgressEstimator.
        """
        from place_layers.search_strategy.cost_constrained.hierarchical import (
            allocation_bounds,
        )

        total_lower, total_upper = 0, 0
        allocation_info = []

        for allocation in allocations:
            budgets = phase_budgets_fn(allocation)
            lower, upper = allocation_bounds(
                budgets=budgets,
                num_layers=num_layers,
                beam_width=beam_width,
                cross_phase_beam_width=cross_phase_beam_width,
            )
            total_lower += lower
            total_upper += upper
            allocation_info.append(
                AllocationProgressInfo(
                    allocation=allocation,
                    budgets=budgets,
                    estimated_lower=lower,
                    estimated_upper=upper,
                )
            )

        return SearchProgressEstimator(
            initial=ProgressEstimate(lower=total_lower, upper=total_upper),
            allocation_info=allocation_info,
        )


# Discriminated union for Hydra
ProgressEstimatorConfig = Annotated[
    HierarchicalProgressEstimatorConfig,
    Field(discriminator="estimator_type"),
]
