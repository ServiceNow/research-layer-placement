"""Shared test helpers for place_layers tests."""

from place_layers.evaluator.evaluators import AggregationStrategy, Evaluators


def wrap_evaluator(evaluator, name: str = "main") -> Evaluators:
    """Wrap a single evaluator in an Evaluators container for tests."""
    return Evaluators(
        evaluators={name: evaluator},
        weights={name: 1.0},
        aggregation=AggregationStrategy.FIRST_ONLY,
        num_layers=evaluator.num_layers,
        mixer_names=evaluator.mixer_names,
    )
