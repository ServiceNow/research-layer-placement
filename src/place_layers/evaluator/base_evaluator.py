"""Abstract base class for parameter evaluators."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Self


if TYPE_CHECKING:
    from place_layers.evaluator.base_evaluator_config import BaseEvaluatorConfig


class BaseEvaluator[ParametersT]:
    """Abstract base class for parameter evaluators.

    Evaluators should be used as async context managers to ensure proper
    resource cleanup:

        async with config.create_evaluator() as evaluator:
            score = await evaluator.evaluate(placement)
    """

    def __init__(self, config: BaseEvaluatorConfig) -> None:
        """
        Initialize the evaluator with configuration.

        Args:
            config: Evaluator configuration.
        """
        self.config = config

    async def __aenter__(self) -> Self:
        """Enter async context (default: no-op)."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context and cleanup resources (default: no-op)."""
        pass

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of layers in the model.

        Implementations should derive this from the actual model or config.
        """
        ...

    @property
    @abstractmethod
    def num_mixers(self) -> int:
        """Number of mixer types per layer.

        Implementations should derive this from the actual model or config.
        """
        ...

    @property
    @abstractmethod
    def mixer_names(self) -> tuple[str, ...]:
        """Available mixer types (e.g., ('attention', 'gdn', 'kda')).

        Implementations should derive this from the actual model or backend.
        """
        ...

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
