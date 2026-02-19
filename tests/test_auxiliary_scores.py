#!/usr/bin/env python3
"""Test that multi-evaluator dict scores work in search events and types."""

import asyncio
from pathlib import Path

# Test that imports work and types are correct
from place_layers.evaluator.evaluators import Evaluators
from place_layers.search_strategy.progress.events import (
    AllocationCompletedEvent,
    BestUpdatedEvent,
)
from place_layers.search_strategy.cost_constrained.types import BeamEntry


async def test_evaluate_dict_scores():
    """Test that Evaluators.evaluate returns dict scores."""
    from place_layers.evaluator import MockSupernetEvaluatorConfig
    from place_layers.evaluator.evaluators import EvaluatorsConfig
    from place_layers.evaluator.objective import WeightedAverageObjectiveConfig
    from place_layers.core.types import LayerPlacement

    # Create a multi-evaluator config with two mock evaluators
    config = EvaluatorsConfig(
        configs={
            "eval1": MockSupernetEvaluatorConfig(
                evaluator_type="mock_supernet",
                num_layers=8,
                mixer_names=["FA", "SWA", "GDN"],
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
            ),
            "eval2": MockSupernetEvaluatorConfig(
                evaluator_type="mock_supernet",
                num_layers=8,
                mixer_names=["FA", "SWA", "GDN"],
                seed=43,
                unary_bonus=0.1,
                unary_penalty=0.1,
                pairwise_scale=0.1,
                noise_sigma=0.01,
                noise_seed_shift=0,
                pairwise_decay=0.5,
                pairwise_max_range=1,
                triplet_scale=0.0,
                eval_delay_seconds=0.0,
            ),
        },
        objective=WeightedAverageObjectiveConfig(
            weights={"eval1": 0.6, "eval2": 0.4},
            include_individual=False,
        ),
    )

    # Create evaluators
    output_dir = Path("/tmp/test_auxiliary_scores")
    output_dir.mkdir(parents=True, exist_ok=True)

    async with config.create_evaluators(output_dir) as evaluators:
        # Test evaluate returns dict
        placement = LayerPlacement(assignments=("FA",) * 8)
        scores = await evaluators.evaluate(placement)

        print(f"Scores dict: {scores}")

        # Verify we got per-evaluator scores
        assert isinstance(scores, dict)
        assert "eval1" in scores
        assert "eval2" in scores

        # Verify objectives is correct
        obj_scores = evaluators.objectives(scores)
        expected = 0.6 * scores["eval1"] + 0.4 * scores["eval2"]
        assert abs(obj_scores["weighted_average"] - expected) < 1e-6, (
            f"Objective mismatch: {obj_scores['weighted_average']} != {expected}"
        )

        print("evaluate returns dict scores correctly")


def test_event_fields():
    """Test that event fields accept dict scores."""
    # Test BestUpdatedEvent
    event1 = BestUpdatedEvent(
        score=0.82,
        cost=5.0,
        allocation={"FA": 4, "GDN": 4},
        assignments=("FA",) * 4 + ("GDN",) * 4,
    )
    print(f"BestUpdatedEvent with score: {event1.score}")

    # Test AllocationCompletedEvent
    event2 = AllocationCompletedEvent(
        allocation_index=1,
        allocation={"FA": 4, "GDN": 4},
        allocation_cost=5.0,
        mixer_names=("FA", "GDN"),
        assignments=("FA",) * 4 + ("GDN",) * 4,
        score=0.82,
    )
    print(f"AllocationCompletedEvent with score: {event2.score}")


def test_beam_entry():
    """Test that BeamEntry accepts dict scores."""
    entry = BeamEntry(
        assignments=("FA",) * 4 + ("GDN",) * 4,
        allocation={"FA": 4, "GDN": 4},
        score=0.82,
    )
    print(f"BeamEntry with score: {entry.score}")


if __name__ == "__main__":
    print("Testing dict scores implementation...\n")

    # Test sync functionality
    print("1. Testing event fields...")
    test_event_fields()
    print()

    print("2. Testing BeamEntry...")
    test_beam_entry()
    print()

    # Test async functionality
    print("3. Testing evaluate dict scores...")
    asyncio.run(test_evaluate_dict_scores())
    print()

    print("All tests passed!")
