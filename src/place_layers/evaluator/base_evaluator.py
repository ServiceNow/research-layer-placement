"""Abstract base class for parameter evaluators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

from place_layers.core.types import ParametersT

if TYPE_CHECKING:
    from place_layers.evaluator.base_evaluator_config import BaseEvaluatorConfig


class BaseEvaluator(Generic[ParametersT]):
    """Abstract base class for parameter evaluators."""

    def __init__(self, config: "BaseEvaluatorConfig") -> None:
        """
        Initialize the evaluator with configuration.

        Args:
            config: Evaluator configuration.
        """
        self.config = config

    async def evaluate(self, parameters: ParametersT) -> float:
        """
        Evaluate parameters and return a score.

        Args:
            parameters: The parameters to evaluate.

        Returns:
            The score (higher is better by default).
        """
        raise NotImplementedError

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
        scores = []
        for parameters in parameters_list:
            score = await self.evaluate(parameters)
            scores.append(score)
        return scores
