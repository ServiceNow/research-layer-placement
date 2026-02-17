"""Abstract base class for optimization problems."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from place_layers.evaluator.evaluators import Evaluators
    from place_layers.problem.base_problem_config import BaseProblemConfig


class BaseProblem[ParametersT]:
    """
    Abstract base class for optimization problems.

    A Problem represents an optimization task on a model/evaluator:
    - Owns the Evaluators container (wraps 1+ evaluators with aggregation)
    - Defines constraints (e.g., budget limits)
    - Exposes evaluation API for search algorithms
    """

    def __init__(
        self, evaluators: Evaluators[ParametersT], config: BaseProblemConfig
    ) -> None:
        """
        Initialize the problem with evaluators and configuration.

        Args:
            evaluators: Evaluators container (wraps 1+ evaluators).
            config: Problem configuration (constraints, etc.).
        """
        self.evaluators = evaluators
        self.config = config

    async def evaluate(self, parameters: ParametersT) -> dict[str, float]:
        """
        Evaluate parameters.

        This is the primary API that search algorithms use.
        Delegates to the Evaluators container.

        Args:
            parameters: Parameters to evaluate.

        Returns:
            Dict mapping evaluator name to score.
        """
        return await self.evaluators.evaluate(parameters)

    async def evaluate_batch(self, parameters_list: list[ParametersT]) -> list[dict[str, float]]:
        """
        Evaluate multiple parameter settings in batch.

        Args:
            parameters_list: List of parameters to evaluate.

        Returns:
            List of score dicts in the same order as parameters_list.
        """
        return [await self.evaluate(p) for p in parameters_list]
