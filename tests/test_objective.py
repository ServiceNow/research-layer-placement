"""Tests for the objective hierarchy."""

import pytest
from pydantic import ValidationError

from place_layers.evaluator.objective import (
    IndependentObjectiveConfig,
    MaxObjectiveConfig,
    MinObjectiveConfig,
    ObjectiveConfig,
    WeightedAverageObjectiveConfig,
)
from place_layers.evaluator.evaluators import EvaluatorsConfig
from place_layers.evaluator import MockSupernetEvaluatorConfig


NAMES = ("eval1", "eval2", "eval3")

MOCK_CFG = dict(
    evaluator_type="mock_supernet",
    num_layers=4,
    mixer_names=["FA", "GDN"],
    seed=42,
    unary_bonus=0.1,
    unary_penalty=0.1,
    pairwise_scale=0.1,
    noise_sigma=0.01,
    noise_seed_shift=0,
    pairwise_decay=0.5,
    pairwise_max_range=1,
    triplet_scale=0.0,
    eval_delay_seconds=0.0,
)


# ---------------------------------------------------------------------------
# WeightedAverage
# ---------------------------------------------------------------------------


def test_weighted_average_basic():
    cfg = WeightedAverageObjectiveConfig(
        weights={"eval1": 0.5, "eval2": 0.3, "eval3": 0.2},
        include_individual=False,
    )
    obj = cfg.create_objective(NAMES)

    assert obj.objective_names == ("weighted_average",)
    assert obj.num_objectives == 1
    assert obj.required_evaluators() == frozenset(NAMES)

    scores = {"eval1": 0.8, "eval2": 0.6, "eval3": 0.4}
    result = obj.objectives(scores)
    expected = 0.5 * 0.8 + 0.3 * 0.6 + 0.2 * 0.4
    assert result == {"weighted_average": pytest.approx(expected)}


def test_weighted_average_uniform():
    cfg = WeightedAverageObjectiveConfig(include_individual=False)
    obj = cfg.create_objective(("single",))

    assert obj.objective_names == ("weighted_average",)
    result = obj.objectives({"single": 0.75})
    assert result == {"weighted_average": pytest.approx(0.75)}


def test_weighted_average_include_individual():
    cfg = WeightedAverageObjectiveConfig(
        weights={"eval1": 0.6, "eval2": 0.4},
        include_individual=True,
    )
    names = ("eval1", "eval2")
    obj = cfg.create_objective(names)

    # objective_names = aggregate + individuals
    assert obj.objective_names == ("weighted_average", "eval1", "eval2")
    assert obj.num_objectives == 3

    # required_evaluators for aggregate = all
    assert obj.required_evaluators("weighted_average") == frozenset(names)

    # required_evaluators for individual = just that evaluator
    assert obj.required_evaluators("eval1") == frozenset({"eval1"})
    assert obj.required_evaluators("eval2") == frozenset({"eval2"})

    # required_evaluators(None) = all
    assert obj.required_evaluators() == frozenset(names)

    # objectives() returns aggregate + individuals
    scores = {"eval1": 0.8, "eval2": 0.6}
    result = obj.objectives(scores)
    assert result == {
        "weighted_average": pytest.approx(0.6 * 0.8 + 0.4 * 0.6),
        "eval1": 0.8,
        "eval2": 0.6,
    }


# ---------------------------------------------------------------------------
# Min
# ---------------------------------------------------------------------------


def test_min_basic():
    cfg = MinObjectiveConfig(include_individual=False)
    obj = cfg.create_objective(NAMES)

    assert obj.objective_names == ("min",)
    assert obj.required_evaluators() == frozenset(NAMES)

    scores = {"eval1": 0.8, "eval2": 0.3, "eval3": 0.5}
    assert obj.objectives(scores) == {"min": 0.3}


def test_min_include_individual():
    cfg = MinObjectiveConfig(include_individual=True)
    names = ("a", "b")
    obj = cfg.create_objective(names)

    assert obj.objective_names == ("min", "a", "b")
    assert obj.required_evaluators("a") == frozenset({"a"})
    assert obj.required_evaluators("min") == frozenset(names)

    scores = {"a": 0.9, "b": 0.1}
    result = obj.objectives(scores)
    assert result == {"min": 0.1, "a": 0.9, "b": 0.1}


# ---------------------------------------------------------------------------
# Max
# ---------------------------------------------------------------------------


def test_max_basic():
    cfg = MaxObjectiveConfig(include_individual=False)
    obj = cfg.create_objective(NAMES)

    assert obj.objective_names == ("max",)
    scores = {"eval1": 0.2, "eval2": 0.9, "eval3": 0.5}
    assert obj.objectives(scores) == {"max": 0.9}


def test_max_include_individual():
    cfg = MaxObjectiveConfig(include_individual=True)
    names = ("x", "y")
    obj = cfg.create_objective(names)

    assert obj.objective_names == ("max", "x", "y")
    assert obj.required_evaluators("y") == frozenset({"y"})

    scores = {"x": 0.3, "y": 0.7}
    result = obj.objectives(scores)
    assert result == {"max": 0.7, "x": 0.3, "y": 0.7}


# ---------------------------------------------------------------------------
# Independent (unchanged — no include_individual)
# ---------------------------------------------------------------------------


def test_independent_unchanged():
    cfg = IndependentObjectiveConfig()
    obj = cfg.create_objective(("a", "b"))

    assert obj.objective_names == ("a", "b")
    assert obj.required_evaluators("a") == frozenset({"a"})
    assert obj.required_evaluators() == frozenset({"a", "b"})
    assert obj.objectives({"a": 0.5, "b": 0.9}) == {"a": 0.5, "b": 0.9}

    # IndependentObjectiveConfig should not have include_individual
    assert not hasattr(IndependentObjectiveConfig, "include_individual") or \
        "include_individual" not in IndependentObjectiveConfig.model_fields


# ---------------------------------------------------------------------------
# FirstOnly removed
# ---------------------------------------------------------------------------


def test_first_only_removed():
    """objective_type='first_only' is no longer valid in the union."""
    with pytest.raises(ValidationError):
        from pydantic import TypeAdapter
        ta = TypeAdapter(ObjectiveConfig)
        ta.validate_python({"objective_type": "first_only"})


# ---------------------------------------------------------------------------
# Name collision validation
# ---------------------------------------------------------------------------


def test_name_collision_raises():
    """Evaluator named same as objective_type + include_individual=True -> error."""
    with pytest.raises(ValidationError, match="collides with"):
        EvaluatorsConfig(
            configs={
                "weighted_average": MockSupernetEvaluatorConfig(**MOCK_CFG),
                "other": MockSupernetEvaluatorConfig(**MOCK_CFG),
            },
            objective=WeightedAverageObjectiveConfig(
                include_individual=True,
                weights={"weighted_average": 0.5, "other": 0.5},
            ),
        )


def test_no_collision_when_disabled():
    """Same setup but include_individual=False is fine."""
    cfg = EvaluatorsConfig(
        configs={
            "weighted_average": MockSupernetEvaluatorConfig(**MOCK_CFG),
            "other": MockSupernetEvaluatorConfig(**MOCK_CFG),
        },
        objective=WeightedAverageObjectiveConfig(
            include_individual=False,
            weights={"weighted_average": 0.5, "other": 0.5},
        ),
    )
    assert cfg.objective.objective_type == "weighted_average"
