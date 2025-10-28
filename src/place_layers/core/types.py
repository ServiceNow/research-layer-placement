"""Core type definitions for the optimization framework."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class SearchResult[ParametersT](BaseModel):
    """Complete results from an optimization run."""

    best_parameters: ParametersT
    best_score: float
    total_evaluations: int


class Evaluator[ParametersT](Protocol):
    """Protocol for evaluating parameters."""

    async def evaluate(self, parameters: ParametersT) -> float:
        """
        Evaluate parameters and return a score.

        Args:
            parameters: The parameters to evaluate.

        Returns:
            The score (higher is better by default).
        """
        ...

    async def evaluate_batch(self, parameters_list: list[ParametersT]) -> list[float]:
        """
        Evaluate multiple parameter settings in batch.

        Default implementation evaluates sequentially.
        Subclasses may override for parallel evaluation.

        Args:
            parameters_list: List of parameters to evaluate.

        Returns:
            List of scores in the same order as parameters_list.
        """
        ...
