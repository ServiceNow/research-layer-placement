"""Objective hierarchy: maps N evaluator scores to M objective scores.

Search strategies call ``Evaluators.objectives()`` to map cached evaluator
scores to objective scores, or ``Evaluators.evaluate()`` to run all
evaluators.  ``evaluate_objectives()`` selectively runs only the evaluators
needed for specific objectives.

Four concrete objectives are provided:

- **WeightedAverage** (N->1): weighted sum of all evaluator scores.
- **Min** (N->1): minimum across all evaluator scores.
- **Max** (N->1): maximum across all evaluator scores.
- **Independent** (N->N): each evaluator is its own objective.

The three aggregating objectives (WeightedAverage, Min, Max) support an
``include_individual`` flag that, when True, exposes individual evaluator
scores as additional objectives alongside the aggregate.
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


class AggregatingObjective(BaseObjective):
    """Base for objectives that reduce N evaluator scores to 1 aggregate."""

    config: "AggregatingObjectiveConfig"

    def _aggregate(self, scores: dict[str, float]) -> float:
        raise NotImplementedError

    @property
    def objective_names(self) -> tuple[str, ...]:
        base = (self.config.objective_type,)
        if self.config.include_individual:
            return base + self._evaluator_names
        return base

    def required_evaluators(self, objective_name: str | None = None) -> frozenset[str]:
        if objective_name is None:
            return frozenset(self._evaluator_names)
        if self.config.include_individual and objective_name in self._evaluator_names:
            return frozenset({objective_name})
        return frozenset(self._evaluator_names)

    def objectives(self, scores: dict[str, float]) -> dict[str, float]:
        result = {self.config.objective_type: self._aggregate(scores)}
        if self.config.include_individual:
            result.update(scores)
        return result


class WeightedAverageObjective(AggregatingObjective):
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

    def _aggregate(self, scores: dict[str, float]) -> float:
        return sum(self._weights[k] * scores[k] for k in scores)


class MinObjective(AggregatingObjective):
    """Minimum across all evaluator scores."""

    def _aggregate(self, scores: dict[str, float]) -> float:
        return min(scores.values())


class MaxObjective(AggregatingObjective):
    """Maximum across all evaluator scores."""

    def _aggregate(self, scores: dict[str, float]) -> float:
        return max(scores.values())


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


class AggregatingObjectiveConfig(BaseObjectiveConfig):
    """Base config for aggregating objectives (weighted_average, min, max)."""

    include_individual: bool = Field(
        description="When True, also expose individual evaluator scores as separate objectives.",
    )


class WeightedAverageObjectiveConfig(AggregatingObjectiveConfig):
    _impl_class: ClassVar[type[BaseObjective]] = WeightedAverageObjective

    objective_type: Literal["weighted_average"] = "weighted_average"
    weights: dict[str, float] = Field(
        default_factory=dict,
        description="Per-evaluator weights. Empty = uniform (auto-filled at creation).",
    )


class MinObjectiveConfig(AggregatingObjectiveConfig):
    _impl_class: ClassVar[type[BaseObjective]] = MinObjective

    objective_type: Literal["min"] = "min"


class MaxObjectiveConfig(AggregatingObjectiveConfig):
    _impl_class: ClassVar[type[BaseObjective]] = MaxObjective

    objective_type: Literal["max"] = "max"


class IndependentObjectiveConfig(BaseObjectiveConfig):
    _impl_class: ClassVar[type[BaseObjective]] = IndependentObjective

    objective_type: Literal["independent"] = "independent"


# Discriminated union
ObjectiveConfig = Annotated[
    WeightedAverageObjectiveConfig
    | MinObjectiveConfig
    | MaxObjectiveConfig
    | IndependentObjectiveConfig,
    Field(discriminator="objective_type"),
]
