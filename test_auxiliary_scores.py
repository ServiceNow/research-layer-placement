#!/usr/bin/env python3
"""Quick test to verify auxiliary scores work in search events."""

import asyncio
from pathlib import Path

# Test that imports work and types are correct
from place_layers.evaluator.multi_evaluator_config import EvaluatorInstances
from place_layers.search_strategy.progress.events import (
    AllocationCompletedEvent,
    BestUpdatedEvent,
)
from place_layers.search_strategy.cost_constrained.types import BeamEntry


async def test_evaluate_with_details():
    """Test that EvaluatorInstances.evaluate_with_details works."""
    from place_layers.evaluator import MockSupernetEvaluatorConfig
    from place_layers.evaluator.multi_evaluator_config import (
        AggregationStrategy,
        EvaluatorsConfig,
    )
    from place_layers.core.types import LayerPlacement

    # Create a multi-evaluator config with two mock evaluators
    config = EvaluatorsConfig(
        evaluators_dict={
            "eval1": MockSupernetEvaluatorConfig(
                evaluator_type="mock_supernet",
                num_layers=8,
                mixer_names=["FA", "SWA", "GDN"],
                base_score=0.5,
                seed=42,
                unary_bonus=0.1,
                unary_penalty=0.1,
                pairwise_scale=0.1,
                noise_sigma=0.01,
                eval_delay_seconds=0.0,
            ),
            "eval2": MockSupernetEvaluatorConfig(
                evaluator_type="mock_supernet",
                num_layers=8,
                mixer_names=["FA", "SWA", "GDN"],
                base_score=0.3,
                seed=43,
                unary_bonus=0.1,
                unary_penalty=0.1,
                pairwise_scale=0.1,
                noise_sigma=0.01,
                eval_delay_seconds=0.0,
            ),
        },
        weights={"eval1": 0.6, "eval2": 0.4},
        aggregation=AggregationStrategy.WEIGHTED_AVERAGE,
    )

    # Create evaluators
    output_dir = Path("/tmp/test_auxiliary_scores")
    output_dir.mkdir(parents=True, exist_ok=True)

    async with config.create_evaluators(output_dir) as evaluators:
        # Test evaluate_with_details
        placement = LayerPlacement(assignments=("FA",) * 8)
        score, auxiliary = await evaluators.evaluate_with_details(placement)

        print(f"✓ Aggregated score: {score:.4f}")
        print(f"✓ Auxiliary scores: {auxiliary}")

        # Verify we got auxiliary scores
        assert auxiliary is not None, "Should have auxiliary scores with 2 evaluators"
        assert "eval1" in auxiliary
        assert "eval2" in auxiliary

        # Verify aggregation is correct
        expected = 0.6 * auxiliary["eval1"] + 0.4 * auxiliary["eval2"]
        assert abs(score - expected) < 1e-6, f"Score mismatch: {score} != {expected}"

        print("✓ evaluate_with_details works correctly")


def test_event_fields():
    """Test that event fields exist and accept auxiliary_scores."""
    # Test BestUpdatedEvent
    event1 = BestUpdatedEvent(
        score=0.8,
        cost=5.0,
        allocation={"FA": 4, "GDN": 4},
        assignments=("FA",) * 4 + ("GDN",) * 4,
        auxiliary_scores={"eval1": 0.9, "eval2": 0.7},
    )
    print(f"✓ BestUpdatedEvent with auxiliary_scores: {event1.auxiliary_scores}")

    # Test AllocationCompletedEvent
    event2 = AllocationCompletedEvent(
        allocation_index=1,
        allocation={"FA": 4, "GDN": 4},
        allocation_cost=5.0,
        mixer_names=("FA", "GDN"),
        assignments=("FA",) * 4 + ("GDN",) * 4,
        score=0.8,
        auxiliary_scores={"eval1": 0.9, "eval2": 0.7},
    )
    print(f"✓ AllocationCompletedEvent with auxiliary_scores: {event2.auxiliary_scores}")


def test_beam_entry():
    """Test that BeamEntry accepts auxiliary_scores."""
    entry = BeamEntry(
        assignments=("FA",) * 4 + ("GDN",) * 4,
        allocation={"FA": 4, "GDN": 4},
        score=0.8,
        auxiliary_scores={"eval1": 0.9, "eval2": 0.7},
    )
    print(f"✓ BeamEntry with auxiliary_scores: {entry.auxiliary_scores}")


if __name__ == "__main__":
    print("Testing auxiliary scores implementation...\n")

    # Test sync functionality
    print("1. Testing event fields...")
    test_event_fields()
    print()

    print("2. Testing BeamEntry...")
    test_beam_entry()
    print()

    # Test async functionality
    print("3. Testing evaluate_with_details...")
    asyncio.run(test_evaluate_with_details())
    print()

    print("✅ All tests passed!")
