"""Shared types for cost-constrained search strategies."""

from typing import Literal

from pydantic import BaseModel

from place_layers.core.types import LayerPlacement, SearchResult


class BeamEntry(BaseModel):
    """An entry in the search beam representing an assignment.

    Note: The 'cost' is intentionally NOT stored here. Always compute it
    fresh via cost_model.allocation_cost(allocation) to avoid stale values.
    """

    model_config = {"frozen": True}

    assignments: tuple[str, ...]  # Mixer names, one per layer
    allocation: dict[str, int]  # Counts per mixer, keyed by mixer name
    score: float | None = None  # None until evaluated
    auxiliary_scores: dict[str, float] | None = None  # Individual scores from multi-evaluator


class CostConstrainedSearchResult(SearchResult[LayerPlacement]):
    """Search results with cost and constraint information."""

    final_cost: float
    mode: Literal["max_performance", "min_cost"]
    constraint_value: float  # cost_budget or performance_target
    allocations_explored: int
