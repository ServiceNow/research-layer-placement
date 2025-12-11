"""Shared types for cost-constrained search strategies."""

from typing import Literal

from pydantic import BaseModel

from place_layers.core.types import LayerPlacement, SearchResult


class BeamEntry(BaseModel):
    """An entry in the search beam representing an assignment."""

    model_config = {"frozen": True}

    assignments: tuple[int, ...]
    allocation: tuple[int, ...]  # Counts per mixer
    cost: float
    score: float | None = None  # None until evaluated


class CostConstrainedSearchResult(SearchResult[LayerPlacement]):
    """Search results with cost and constraint information."""

    final_cost: float
    mode: Literal["max_performance", "min_cost"]
    constraint_value: float  # cost_budget or performance_target
    allocations_explored: int
