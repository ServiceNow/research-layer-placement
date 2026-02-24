"""Abstract base class for optimization problems."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from place_layers.evaluator.evaluators import Evaluators
    from place_layers.problem.base_problem_config import BaseProblemConfig


class BaseProblem[ParametersT]:
    """
    Abstract base class for optimization problems.

    A Problem represents a scalar optimization task: one objective score
    derived from 1+ evaluators via an objective function.  Search strategies
    call ``evaluate()`` and receive a float — they are pure scalar optimizers.

    Multi-objective = M problems, M sequential searches.
    """

    def __init__(
        self,
        evaluators: Evaluators[ParametersT],
        config: BaseProblemConfig,
        objective_name: str,
    ) -> None:
        """
        Initialize the problem with evaluators and configuration.

        Args:
            evaluators: Evaluators container (wraps 1+ evaluators).
            config: Problem configuration (constraints, etc.).
            objective_name: Which objective score this problem optimizes.
        """
        self.evaluators = evaluators
        self.config = config
        self.objective_name = objective_name
        self._cache: dict[ParametersT, float] = {}

    @property
    def num_layers(self) -> int:
        """Number of layers (delegated from evaluators)."""
        return self.evaluators.num_layers

    async def evaluate(self, parameters: ParametersT) -> float:
        """
        Evaluate parameters and return this problem's objective score.

        This is the primary API that search algorithms use.
        Results are cached: repeated evaluations of the same parameters are free.

        Args:
            parameters: Parameters to evaluate.

        Returns:
            Scalar objective score for this problem's objective_name.
        """
        if parameters in self._cache:
            return self._cache[parameters]
        result = await self.evaluators.evaluate_objectives(
            parameters, names=[self.objective_name]
        )
        score = result[self.objective_name]
        self._cache[parameters] = score
        return score

    async def evaluate_batch(self, parameters_list: list[ParametersT]) -> list[float]:
        """
        Evaluate multiple parameter settings in batch.

        Args:
            parameters_list: List of parameters to evaluate.

        Returns:
            List of scalar scores in the same order as parameters_list.
        """
        return [await self.evaluate(p) for p in parameters_list]
