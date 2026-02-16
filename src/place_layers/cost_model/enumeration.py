"""Allocation enumeration utilities for cost-constrained search."""

from __future__ import annotations

from place_layers.cost_model.base_cost_model import Allocation, BaseCostModel


def enumerate_feasible_allocations(
    cost_model: BaseCostModel,
    num_layers: int,
    cost_budget: float | None,
    cost_floor: float | None = None,
) -> list[Allocation]:
    """Collect all allocations within budget (or all if no budget).

    Args:
        cost_model: The cost model to query.
        num_layers: Number of layers.
        cost_budget: If set, only include allocations with cost <= budget.
        cost_floor: If set, only include allocations with cost >= floor.

    Returns:
        List of feasible allocations.
    """
    feasible: list[Allocation] = []
    for alloc_cost, allocation in cost_model.allocations_by_cost(
        num_layers, direction="increasing"
    ):
        if cost_budget is not None and alloc_cost > cost_budget:
            break  # All subsequent allocations are more expensive
        if cost_floor is not None and alloc_cost < cost_floor - 1e-9:
            continue  # Skip allocations below the floor
        feasible.append(allocation)
    return feasible
