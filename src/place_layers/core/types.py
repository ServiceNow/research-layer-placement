"""Core type definitions for the optimization framework."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LayerPlacement(BaseModel):
    """
    Immutable assignment of layers to mixer types.

    This is the primary parameter type for layer placement optimization.
    Each layer is assigned to exactly one mixer type.
    """

    model_config = {"frozen": True}

    assignments: tuple[int, ...] = Field(
        description="Mapping from layer index to mixer index"
    )


class SearchResult[ParametersT](BaseModel):
    """Complete results from an optimization run."""

    best_parameters: ParametersT
    best_score: float
    total_evaluations: int
