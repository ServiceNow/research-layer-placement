"""Shared test helpers for place_layers tests."""

from place_layers.evaluator.evaluators import Evaluators
from place_layers.evaluator.objective import WeightedAverageObjectiveConfig


def make_evaluators(evaluator, name: str = "main") -> Evaluators:
    """Wrap a single evaluator in an Evaluators container for tests."""
    names = (name,)
    return Evaluators(
        evaluators={name: evaluator},
        num_layers=evaluator.num_layers,
        mixer_names=evaluator.mixer_names,
        objective=WeightedAverageObjectiveConfig().create_objective(names),
    )


# Backward-compatible alias
wrap_evaluator = make_evaluators
