"""Core type definitions for the optimization framework."""

from __future__ import annotations

from pydantic import BaseModel


class SearchResult[ParametersT](BaseModel):
    """Complete results from an optimization run."""

    best_parameters: ParametersT
    best_score: float
    total_evaluations: int
