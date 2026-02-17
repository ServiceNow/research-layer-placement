"""Objective hierarchy: maps N evaluator scores to M objective scores.

Search strategies call ``Evaluators.objectives()`` to map cached evaluator
scores to objective scores, or ``Evaluators.evaluate()`` to run all
evaluators.  ``evaluate_objectives()`` selectively runs only the evaluators
needed for specific objectives.

Five concrete objectives are provided:

- **WeightedAverage** (N->1): weighted sum of all evaluator scores.
- **FirstOnly** (N->1): first evaluator's score only.
- **Min** (N->1): minimum across all evaluator scores.
- **Max** (N->1): maximum across all evaluator scores.
- **Independent** (N->N): each evaluator is its own objective.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Runtime classes
# ---------------------------------------------------------------------------


class BaseObjective:
    """Maps N evaluator scores to M objective scores (1 <= M <= N)."""

    def __init__(
        self,
        config: "BaseObjectiveConfig",
        evaluator_names: tuple[str, ...],
    ) -> None:
        self.config = config
        self._evaluator_names = evaluator_names

    @property
    def objective_names(self) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    def num_objectives(self) -> int:
        return len(self.objective_names)

    def required_evaluators(self, objective_name: str | None = None) -> frozenset[str]:
        """Which evaluators are needed to compute an objective.

        Args:
            objective_name: Specific objective.  ``None`` = all evaluators
                needed by all objectives combined.

        Consumed internally by ``Evaluators.evaluate_objectives()`` to
        determine which evaluators to run.  Search strategies never call
        this directly.
        """
        raise NotImplementedError

    def objectives(self, scores: dict[str, float]) -> dict[str, float]:
        """Map evaluator scores to objective scores."""
        raise NotImplementedError


class WeightedAverageObjective(BaseObjective):
    """N evaluator scores -> 1 objective via weighted sum."""

    def __init__(
        self,
        config: "WeightedAverageObjectiveConfig",
        evaluator_names: tuple[str, ...],
    ) -> None:
        super().__init__(config, evaluator_names)
        if config.weights:
            self._weights = config.weights
        else:
            n = len(evaluator_names)
            self._weights = {name: 1.0 / n for name in evaluator_names}

    @property
    def objective_names(self) -> tuple[str, ...]:
        return (self.config.objective_type,)

    def required_evaluators(self, objective_name: str | None = None) -> frozenset[str]:
        return frozenset(self._evaluator_names)

    def objectives(self, scores: dict[str, float]) -> dict[str, float]:
        name = self.config.objective_type
        return {name: sum(self._weights[k] * scores[k] for k in scores)}


class FirstOnlyObjective(BaseObjective):
    """Use first evaluator's score only (others are auxiliary metrics)."""

    @property
    def objective_names(self) -> tuple[str, ...]:
        return (self.config.objective_type,)

    def required_evaluators(self, objective_name: str | None = None) -> frozenset[str]:
        return frozenset({self._evaluator_names[0]})

    def objectives(self, scores: dict[str, float]) -> dict[str, float]:
        return {self.config.objective_type: scores[self._evaluator_names[0]]}


class MinObjective(BaseObjective):
    """Minimum across all evaluator scores."""

    @property
    def objective_names(self) -> tuple[str, ...]:
        return (self.config.objective_type,)

    def required_evaluators(self, objective_name: str | None = None) -> frozenset[str]:
        return frozenset(self._evaluator_names)

    def objectives(self, scores: dict[str, float]) -> dict[str, float]:
        return {self.config.objective_type: min(scores.values())}


class MaxObjective(BaseObjective):
    """Maximum across all evaluator scores."""

    @property
    def objective_names(self) -> tuple[str, ...]:
        return (self.config.objective_type,)

    def required_evaluators(self, objective_name: str | None = None) -> frozenset[str]:
        return frozenset(self._evaluator_names)

    def objectives(self, scores: dict[str, float]) -> dict[str, float]:
        return {self.config.objective_type: max(scores.values())}


class IndependentObjective(BaseObjective):
    """Each evaluator is its own objective.  Search solves N problems in parallel."""

    @property
    def objective_names(self) -> tuple[str, ...]:
        return self._evaluator_names

    def required_evaluators(self, objective_name: str | None = None) -> frozenset[str]:
        if objective_name is not None:
            return frozenset({objective_name})
        return frozenset(self._evaluator_names)

    def objectives(self, scores: dict[str, float]) -> dict[str, float]:
        return dict(scores)


# ---------------------------------------------------------------------------
# Config classes (discriminated union)
# ---------------------------------------------------------------------------


class BaseObjectiveConfig(BaseModel):
    """Base config for objectives.  Subclasses set ``_impl_class``."""

    model_config = {"frozen": True}

    objective_type: str = Field(description="Type identifier")
    _impl_class: ClassVar[type[BaseObjective]]

    def create_objective(self, evaluator_names: tuple[str, ...]) -> BaseObjective:
        return self._impl_class(config=self, evaluator_names=evaluator_names)


class WeightedAverageObjectiveConfig(BaseObjectiveConfig):
    _impl_class: ClassVar[type[BaseObjective]] = WeightedAverageObjective

    objective_type: Literal["weighted_average"] = "weighted_average"
    weights: dict[str, float] = Field(
        default_factory=dict,
        description="Per-evaluator weights. Empty = uniform (auto-filled at creation).",
    )


class FirstOnlyObjectiveConfig(BaseObjectiveConfig):
    _impl_class: ClassVar[type[BaseObjective]] = FirstOnlyObjective

    objective_type: Literal["first_only"] = "first_only"


class MinObjectiveConfig(BaseObjectiveConfig):
    _impl_class: ClassVar[type[BaseObjective]] = MinObjective

    objective_type: Literal["min"] = "min"


class MaxObjectiveConfig(BaseObjectiveConfig):
    _impl_class: ClassVar[type[BaseObjective]] = MaxObjective

    objective_type: Literal["max"] = "max"


class IndependentObjectiveConfig(BaseObjectiveConfig):
    _impl_class: ClassVar[type[BaseObjective]] = IndependentObjective

    objective_type: Literal["independent"] = "independent"


# Discriminated union
ObjectiveConfig = Annotated[
    WeightedAverageObjectiveConfig
    | FirstOnlyObjectiveConfig
    | MinObjectiveConfig
    | MaxObjectiveConfig
    | IndependentObjectiveConfig,
    Field(discriminator="objective_type"),
]
