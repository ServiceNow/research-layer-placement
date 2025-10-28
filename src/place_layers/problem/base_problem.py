"""Abstract base class for optimization problems."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic

from place_layers.core.types import ParametersT

if TYPE_CHECKING:
    from place_layers.evaluator.base_evaluator import BaseEvaluator
    from place_layers.problem.base_problem_config import BaseProblemConfig


class BaseProblem(Generic[ParametersT]):
    """
    Abstract base class for optimization problems.

    A Problem represents an optimization task on a model/evaluator:
    - Owns the evaluator (the model being optimized)
    - Defines constraints (e.g., budget limits)
    - Exposes evaluation API for search algorithms
    """

    def __init__(
        self, evaluator: "BaseEvaluator[ParametersT]", config: "BaseProblemConfig"
    ) -> None:
        """
        Initialize the problem with an evaluator and configuration.

        Args:
            evaluator: The model/evaluator being optimized.
            config: Problem configuration (constraints, etc.).
        """
        self.evaluator = evaluator
        self.config = config

    async def evaluate(self, parameters: ParametersT) -> float:
        """
        Evaluate parameters.

        This is the primary API that search algorithms use.
        Delegates to the evaluator.

        Args:
            parameters: Parameters to evaluate.

        Returns:
            Score (higher is better by default).
        """
        return await self.evaluator.evaluate(parameters)

    async def evaluate_batch(self, parameters_list: list[ParametersT]) -> list[float]:
        """
        Evaluate multiple parameter settings in batch.

        Delegates to the evaluator's batch evaluation.

        Args:
            parameters_list: List of parameters to evaluate.

        Returns:
            List of scores in the same order as parameters_list.
        """
        return await self.evaluator.evaluate_batch(parameters_list)
