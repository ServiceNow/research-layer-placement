"""Abstract base class for parameter evaluators."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import ClassVar, Self

from pydantic import BaseModel, Field


class BaseEvaluator[ParametersT]:
    """Abstract base class for parameter evaluators.

    Evaluators should be used as async context managers to ensure proper
    resource cleanup:

        async with config.create_evaluator() as evaluator:
            score = await evaluator.evaluate(placement)
    """

    def __init__(self, config: "BaseEvaluatorConfig") -> None:
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
    def mixer_names(self) -> frozenset[str]:
        """Available mixer types (e.g., {'attention', 'gdn', 'kda'}).

        This is discovered from the model and used only for validation.
        For ordering, use cost_model.phase_order.

        Implementations should derive this from the actual model or backend.
        """
        ...

    @property
    def num_mixers(self) -> int:
        """Number of mixer types."""
        return len(self.mixer_names)

    def get_metrics(self) -> dict[str, float] | None:
        """Return evaluator-specific metrics beyond the main score.

        Subclasses can override to provide additional metrics that will be
        included in ``evaluation_results.json`` under the ``metrics`` key.

        Returns:
            Dictionary of metric name to value, or None if no extra metrics.
        """
        return None

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


class BaseEvaluatorConfig(BaseModel):
    """Base configuration for all evaluator implementations."""

    model_config = {"arbitrary_types_allowed": True}

    evaluator_type: str = Field(description="Type identifier for evaluator")
    _impl_class: ClassVar[type[BaseEvaluator]]

    def create_evaluator(self, output_dir: "Path | None" = None) -> BaseEvaluator:
        """Smart constructor: instantiate evaluator from config.

        Args:
            output_dir: Base output directory for results. Required for
                       evaluators that write output (BackendEvaluator subclasses).
                       Can be None for evaluators that don't need output (e.g., mock).
        """
        return self._impl_class(config=self)
