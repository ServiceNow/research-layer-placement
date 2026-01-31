"""Evaluation count calculator command for hierarchical beam search.

This module provides a CLI command to calculate lower and upper bounds
on the number of evaluations for hierarchical beam search, given problem
parameters.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import ClassVar, Literal

from hydra.core.hydra_config import HydraConfig
from pydantic import Field

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.cost_model.linear_cost_model import LinearCostModelConfig
from place_layers.search_strategy.cost_constrained.hierarchical import (
    allocation_bounds,
    phase_eval_lower,
    phase_eval_upper,
)


logger = logging.getLogger(__name__)


# Re-export bounds functions for backward compatibility
__all__ = [
    "phase_eval_lower",
    "phase_eval_upper",
    "allocation_bounds",
    "calculate_bounds",
    "EvalCountCommand",
    "EvalCountCommandConfig",
]


def calculate_bounds(
    num_layers: int,
    mixer_costs: dict[str, float],
    cost_budget: float,
    max_allocations: int,
    beam_width: int,
    cross_phase_beam_width: int,
) -> dict:
    """
    Calculate total evaluation count bounds across all feasible allocations.

    Args:
        num_layers: Total number of layers (N).
        mixer_costs: Dict mapping mixer names to cost coefficients.
        cost_budget: Maximum cost budget.
        max_allocations: Maximum number of allocations to analyze.
        beam_width: Within-phase beam width (B).
        cross_phase_beam_width: Cross-phase beam width (C).

    Returns:
        Dict with lower_bound, upper_bound, allocations_analyzed, and per-allocation details.
    """
    # Create cost model to enumerate allocations
    config = LinearCostModelConfig(mixer_costs=mixer_costs)
    mixer_names = list(mixer_costs.keys())
    cost_model = config.create_cost_model(mixer_names)

    total_lower, total_upper = 0, 0
    allocations_analyzed = 0
    allocation_details: list[dict] = []

    # Enumerate allocations in decreasing cost order (like the actual search)
    for alloc_cost, allocation in cost_model.allocations_by_cost(
        num_layers, "decreasing"
    ):
        if alloc_cost > cost_budget:
            continue
        if allocations_analyzed >= max_allocations:
            break

        # Phase budgets = allocation counts for first K-1 mixers (in phase order)
        # The phase order follows mixer_costs key order
        phase_order = cost_model.phase_order
        name_to_alloc_idx = {name: i for i, name in enumerate(mixer_names)}

        # Build budgets in phase order (excluding the last mixer which gets remainder)
        budgets = [allocation[name_to_alloc_idx[phase_order[i]]] for i in range(len(phase_order) - 1)]

        lower, upper = allocation_bounds(
            budgets, num_layers, beam_width, cross_phase_beam_width
        )
        total_lower += lower
        total_upper += upper
        allocations_analyzed += 1

        allocation_details.append({
            "allocation": list(allocation),
            "allocation_dict": dict(zip(mixer_names, allocation, strict=True)),
            "cost": alloc_cost,
            "budgets": budgets,
            "lower_bound": lower,
            "upper_bound": upper,
        })

    return {
        "lower_bound": total_lower,
        "upper_bound": total_upper,
        "allocations_analyzed": allocations_analyzed,
        "allocation_details": allocation_details,
    }


class EvalCountCommand(BaseCommand):
    """Calculate evaluation count bounds for hierarchical beam search."""

    config: "EvalCountCommandConfig"

    async def run(self) -> None:
        """Execute evaluation count calculation."""
        # Get output directory from Hydra's runtime config
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)

        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION COUNT CALCULATOR")
        logger.info("=" * 60)

        # Extract parameters
        num_layers = self.config.num_layers
        mixer_costs = dict(self.config.cost_model.mixer_costs)
        cost_budget = self.config.cost_budget
        max_allocations = self.config.max_allocations
        beam_width = self.config.beam_width
        cross_phase_beam_width = self.config.cross_phase_beam_width

        logger.info(f"Number of layers (N): {num_layers}")
        logger.info(f"Number of mixers (K): {len(mixer_costs)}")
        logger.info(f"Mixer costs: {mixer_costs}")
        logger.info(f"Cost budget: {cost_budget}")
        logger.info(f"Max allocations: {max_allocations}")
        logger.info(f"Beam width (B): {beam_width}")
        logger.info(f"Cross-phase beam width (C): {cross_phase_beam_width}")

        # Calculate bounds
        results = calculate_bounds(
            num_layers=num_layers,
            mixer_costs=mixer_costs,
            cost_budget=cost_budget,
            max_allocations=max_allocations,
            beam_width=beam_width,
            cross_phase_beam_width=cross_phase_beam_width,
        )

        # Log results
        logger.info("\n" + "=" * 60)
        logger.info("RESULTS")
        logger.info("=" * 60)
        logger.info(f"Allocations analyzed: {results['allocations_analyzed']}")
        logger.info(f"Lower bound: {results['lower_bound']:,} evaluations")
        logger.info(f"Upper bound: {results['upper_bound']:,} evaluations")

        if results["allocations_analyzed"] > 0:
            avg_lower = results["lower_bound"] / results["allocations_analyzed"]
            avg_upper = results["upper_bound"] / results["allocations_analyzed"]
            logger.info(f"Average per allocation: {avg_lower:.1f} - {avg_upper:.1f}")

        # Log per-allocation breakdown if not too many
        if results["allocations_analyzed"] <= 20:
            logger.info("\n--- Per-Allocation Breakdown ---")
            for i, detail in enumerate(results["allocation_details"]):
                logger.info(
                    f"  {i + 1}. {detail['allocation_dict']} "
                    f"(cost={detail['cost']:.4f}): "
                    f"{detail['lower_bound']:,} - {detail['upper_bound']:,}"
                )

        # Save JSON results
        output = {
            "parameters": {
                "num_layers": num_layers,
                "num_mixers": len(mixer_costs),
                "mixer_costs": mixer_costs,
                "cost_budget": cost_budget,
                "max_allocations": max_allocations,
                "beam_width": beam_width,
                "cross_phase_beam_width": cross_phase_beam_width,
            },
            "results": {
                "lower_bound": results["lower_bound"],
                "upper_bound": results["upper_bound"],
                "allocations_analyzed": results["allocations_analyzed"],
            },
            "allocation_details": results["allocation_details"],
        }

        json_path = output_dir / "eval_count_bounds.json"
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"\nSaved results to: {json_path}")

        logger.info("\n" + "=" * 60)
        logger.info("CALCULATION COMPLETE")
        logger.info("=" * 60 + "\n")


class EvalCountCommandConfig(BaseCommandConfig):
    """Configuration for evaluation count calculator command."""

    _command_class: ClassVar[type[BaseCommand]] = EvalCountCommand

    command: Literal["eval_count"] = "eval_count"

    num_layers: int = Field(
        description="Number of layers in the model (N)",
        gt=0,
    )

    cost_model: LinearCostModelConfig = Field(
        description="Cost model configuration with mixer costs",
    )

    cost_budget: float = Field(
        description="Maximum cost budget for feasible allocations",
        gt=0,
    )

    max_allocations: int = Field(
        description="Maximum number of allocations to analyze",
        gt=0,
    )

    beam_width: int = Field(
        description="Within-phase beam width (B)",
        gt=0,
    )

    cross_phase_beam_width: int = Field(
        description="Cross-phase beam width (C)",
        gt=0,
        default=1,
    )
