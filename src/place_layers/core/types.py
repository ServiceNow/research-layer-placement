"""Core type definitions for the optimization framework."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    model_config = {"frozen": True}

    role: Literal["system", "user", "assistant"] = Field(
        description="Role of the message sender"
    )
    content: str = Field(description="Message content")


class LayerPlacement(BaseModel):
    """
    Immutable assignment of layers to mixer types.

    This is the primary parameter type for layer placement optimization.
    Each layer is assigned to exactly one mixer type.
    """

    model_config = {"frozen": True}

    assignments: tuple[str, ...] = Field(
        description="Mapping from layer index to mixer name"
    )


class SearchResult[ParametersT](BaseModel):
    """Complete results from an optimization run."""

    best_parameters: ParametersT
    best_scores: dict[str, float]
    total_evaluations: int
