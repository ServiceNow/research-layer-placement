"""Calibrate simulated annealing temperature for cost-constrained search.

This command determines the optimal initial temperature for simulated annealing
by sampling the delta distribution and finding the temperature that achieves
a target acceptance ratio.

Theory:
    The acceptance probability for a worsening move with energy delta d < 0
    at temperature T is: P(accept) = exp(d / T)

    We want to find T such that the average acceptance probability over the
    distribution of worsening moves equals our target (typically 0.8 = 80%).

References:
    - Kirkpatrick et al. (1983) "Optimization by Simulated Annealing"
    - Ben-Ameur (2004) "Computing the Initial Temperature of Simulated Annealing"
      https://link.springer.com/article/10.1023/B:COAP.0000044187.23143.bd
"""

import logging
import math
import random
from pathlib import Path
from typing import ClassVar, Literal

from hydra.core.hydra_config import HydraConfig
from pydantic import Field

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.core.types import LayerPlacement
from place_layers.cost_model import CostModelConfig
from place_layers.cost_model.linear_cost_model import (
    LinearCostModel,
    LinearCostModelConfig,
)
from place_layers.evaluator import EvaluatorConfig
from place_layers.problem.layer_placement import (
    CostConstrainedProblem,
    MaxPerformanceProblemConfig,
)

logger = logging.getLogger(__name__)


class CalibrateSACommand(BaseCommand):
    """Calibrate SA temperature by sampling the delta distribution."""

    config: "CalibrateSACommandConfig"

    async def run(self) -> None:
        """Execute SA calibration."""
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)

        logger.info("=" * 70)
        logger.info("SIMULATED ANNEALING TEMPERATURE CALIBRATION")
        logger.info("=" * 70)
        logger.info(f"Target acceptance ratio: {self.config.target_acceptance:.0%}")
        logger.info(f"Number of samples: {self.config.num_samples}")
        logger.info(f"Max steps (for cooling rate): {self.config.max_steps}")

        # Create evaluator and cost model
        evaluator = self.config.evaluator.create_evaluator()
        cost_model_config = self.config.cost_model

        if not isinstance(cost_model_config, LinearCostModelConfig):
            raise ValueError("Calibration currently only supports LinearCostModel")

        cost_model = LinearCostModel(cost_model_config)

        # Create problem
        problem_config = MaxPerformanceProblemConfig(
            cost_model=cost_model_config,
            cost_budget=self.config.cost_budget,
        )
        problem = CostConstrainedProblem(evaluator=evaluator, config=problem_config)

        # Find a representative allocation at the cost bound
        num_layers = evaluator.num_layers
        allocation = cost_model.allocation_at_cost_bound(
            num_layers, self.config.cost_budget, from_above=True
        )

        logger.info(f"\nUsing representative allocation: {allocation}")
        logger.info(f"Allocation cost: {cost_model.allocation_cost(allocation):.4f}")

        # Sample delta distribution
        logger.info(f"\nSampling {self.config.num_samples} random swaps...")
        deltas = await self._sample_deltas(problem, allocation)

        positive = [d for d in deltas if d > 0]
        negative = [d for d in deltas if d < 0]

        logger.info("\nDelta distribution:")
        logger.info(f"  Total samples: {len(deltas)}")
        logger.info(f"  Improving moves: {len(positive)} ({100*len(positive)/len(deltas):.1f}%)")
        logger.info(f"  Worsening moves: {len(negative)} ({100*len(negative)/len(deltas):.1f}%)")

        if positive:
            logger.info(
                f"  Positive deltas: min={min(positive):.4f}, max={max(positive):.4f}, "
                f"avg={sum(positive)/len(positive):.4f}"
            )
        if negative:
            logger.info(
                f"  Negative deltas: min={min(negative):.4f}, max={max(negative):.4f}, "
                f"avg={sum(negative)/len(negative):.4f}"
            )

        if not negative:
            logger.error("No worsening moves found! Cannot calibrate temperature.")
            return

        # Find temperature for target acceptance
        logger.info("\nFinding temperature for target acceptance...")
        initial_temp = self._find_temperature(negative, self.config.target_acceptance)
        cooling_rate = self._compute_cooling_rate(
            initial_temp, self.config.min_temperature, self.config.max_steps
        )

        # Verify
        actual_acceptance = sum(math.exp(d / initial_temp) for d in negative) / len(negative)

        logger.info("\n" + "=" * 70)
        logger.info("CALIBRATION RESULTS")
        logger.info("=" * 70)
        logger.info(f"\n  initial_temperature: {initial_temp:.4f}")
        logger.info(f"  cooling_rate: {cooling_rate:.6f}")
        logger.info(f"  min_temperature: {self.config.min_temperature}")
        logger.info(f"  max_steps: {self.config.max_steps}")
        logger.info(f"\n  Verification: acceptance at T₀ = {actual_acceptance:.1%}")

        # Show table of acceptance ratios
        logger.info("\nTemperature vs acceptance ratio:")
        for target in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            t = self._find_temperature(negative, target)
            logger.info(f"  {target:.0%} acceptance → T = {t:.4f}")

        # Write results to file
        results_path = output_dir / "calibration_results.yaml"
        with open(results_path, "w") as f:
            f.write("# SA Calibration Results\n")
            f.write(f"# Generated for {num_layers} layers\n")
            f.write(f"# Cost budget: {self.config.cost_budget}\n")
            f.write(f"# Target acceptance: {self.config.target_acceptance:.0%}\n")
            f.write("\n")
            f.write("# Copy these values to your search strategy config:\n")
            f.write(f"initial_temperature: {initial_temp:.4f}\n")
            f.write(f"cooling_rate: {cooling_rate:.6f}\n")
            f.write(f"min_temperature: {self.config.min_temperature}\n")
            f.write(f"max_steps: {self.config.max_steps}\n")

        logger.info(f"\nResults saved to: {results_path}")

        # Also output YAML snippet to logs
        logger.info("\n" + "=" * 70)
        logger.info("YAML CONFIG SNIPPET")
        logger.info("=" * 70)
        logger.info(f"""
# Calibrated for {num_layers} layers, target acceptance: {self.config.target_acceptance:.0%}
strategy_type: cost_constrained_annealing
max_allocations: 30
max_steps: {self.config.max_steps}
initial_temperature: {initial_temp:.4f}
cooling_rate: {cooling_rate:.6f}
min_temperature: {self.config.min_temperature}
""")

    async def _sample_deltas(
        self,
        problem: CostConstrainedProblem,
        allocation: tuple[int, ...],
    ) -> list[float]:
        """Sample random allocation-preserving swaps and return delta distribution."""
        random.seed(self.config.seed)

        num_layers = sum(allocation)
        num_mixers = len(allocation)

        # Start from canonical assignment
        current = []
        for mixer_idx, count in enumerate(allocation):
            current.extend([mixer_idx] * count)

        placement = LayerPlacement(assignments=tuple(current))
        current_score = await problem.evaluate(placement)

        deltas = []
        for _ in range(self.config.num_samples):
            # Group layers by mixer
            layers_by_mixer: list[list[int]] = [[] for _ in range(num_mixers)]
            for i, m in enumerate(current):
                layers_by_mixer[m].append(i)

            # Find mixers with layers
            non_empty = [i for i, g in enumerate(layers_by_mixer) if g]
            if len(non_empty) < 2:
                continue

            # Pick two random mixers and swap one layer from each
            m_a, m_b = random.sample(non_empty, 2)
            l_a = random.choice(layers_by_mixer[m_a])
            l_b = random.choice(layers_by_mixer[m_b])

            # Create swapped assignment
            new = current.copy()
            new[l_a], new[l_b] = new[l_b], new[l_a]

            placement = LayerPlacement(assignments=tuple(new))
            new_score = await problem.evaluate(placement)

            deltas.append(new_score - current_score)

        return deltas

    def _find_temperature(
        self,
        negative_deltas: list[float],
        target_acceptance: float,
        t_low: float = 0.001,
        t_high: float = 100.0,
        tolerance: float = 0.001,
        max_iterations: int = 50,
    ) -> float:
        """Binary search for temperature that achieves target acceptance."""
        for _ in range(max_iterations):
            t_mid = (t_low + t_high) / 2

            avg_acceptance = sum(math.exp(d / t_mid) for d in negative_deltas) / len(
                negative_deltas
            )

            if abs(avg_acceptance - target_acceptance) < tolerance:
                return t_mid

            if avg_acceptance < target_acceptance:
                t_low = t_mid
            else:
                t_high = t_mid

        return t_mid

    def _compute_cooling_rate(
        self,
        initial_temp: float,
        min_temp: float,
        max_steps: int,
    ) -> float:
        """Compute cooling rate to reach min_temp after max_steps."""
        return (min_temp / initial_temp) ** (1 / max_steps)


class CalibrateSACommandConfig(BaseCommandConfig):
    """Configuration for SA calibration command."""

    _command_class: ClassVar[type[BaseCommand]] = CalibrateSACommand

    command: Literal["calibrate_sa"] = "calibrate_sa"

    # Problem setup
    evaluator: EvaluatorConfig = Field(description="Evaluator configuration")
    cost_model: CostModelConfig = Field(description="Cost model configuration")
    cost_budget: float = Field(description="Cost budget for the problem", gt=0)

    # Calibration parameters
    target_acceptance: float = Field(
        description="Target acceptance ratio for worsening moves (e.g., 0.8 = 80%)",
        gt=0,
        lt=1,
    )
    num_samples: int = Field(
        description="Number of random swaps to sample",
        gt=0,
    )
    max_steps: int = Field(
        description="Max SA steps per allocation (for computing cooling rate)",
        gt=0,
    )
    min_temperature: float = Field(
        description="Minimum temperature for SA",
        gt=0,
    )
    seed: int = Field(
        description="Random seed for sampling",
    )
