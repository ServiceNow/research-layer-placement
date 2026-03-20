"""Allocation enumeration utilities for cost-constrained search."""

from __future__ import annotations

from place_layers.cost_model.base_cost_model import Allocation, BaseCostModel


def enumerate_feasible_allocations(
    cost_model: BaseCostModel,
    num_layers: int,
    cost_budget: float | None,
    cost_floor: float | None = None,
    min_layers_per_mixer: int = 0,
) -> list[Allocation]:
    """Collect all allocations within budget (or all if no budget).

    Args:
        cost_model: The cost model to query.
        num_layers: Number of layers.
        cost_budget: If set, only include allocations with cost <= budget.
        cost_floor: If set, only include allocations with cost >= floor.
        min_layers_per_mixer: If > 0, exclude allocations where any mixer has
            a nonzero count less than this value. Mixers with 0 layers (unused)
            are always allowed.

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
        if min_layers_per_mixer > 0 and any(
            0 < count < min_layers_per_mixer for count in allocation.values()
        ):
            continue  # Skip allocations where any active mixer has too few layers
        feasible.append(allocation)
    return feasible
