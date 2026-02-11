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
from pydantic import BaseModel, Field

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.core.types import LayerPlacement
from place_layers.cost_model.base_cost_model import Allocation
from place_layers.cost_model.linear_cost_model import LinearCostModel
from place_layers.annealing.moves import (
    generate_move,
    random_assignment,
    sample_perturbation_size,
)
from place_layers.evaluator import EvaluatorConfig
from place_layers.problem import ProblemConfig
from place_layers.problem.layer_placement import (
    CostConstrainedProblem,
    MaxPerformanceProblemConfig,
)
from place_layers.result_handler import ResultHandlerConfig

logger = logging.getLogger(__name__)


class CalibrationResult(BaseModel):
    """Result of SA temperature calibration."""

    model_config = {"frozen": True}

    # Calibrated SA parameters (the main output)
    initial_temperature: float = Field(description="Calibrated initial temperature")
    cooling_rate: float = Field(description="Computed cooling rate")
    min_temperature: float = Field(description="Minimum temperature for SA")
    max_steps: int = Field(description="Max SA steps per allocation")

    # Metadata about the calibration run
    num_layers: int = Field(description="Number of model layers")
    cost_budget: float = Field(description="Cost budget used for calibration")
    target_acceptance: float = Field(description="Target acceptance ratio")
    actual_acceptance: float = Field(description="Achieved acceptance ratio at T0")
    num_samples: int = Field(description="Number of moves sampled")

    # Move operator settings used during calibration
    initial_temperature_guess: float = Field(
        description="Temperature guess used for k-distribution"
    )
    max_ruin_fraction: float = Field(description="Max ruin fraction used")
    contiguous_ruin_probability: float = Field(
        description="Contiguous ruin probability used"
    )

    # Delta distribution summary
    num_deltas: int = Field(description="Total valid (non-no-op) deltas sampled")
    num_improving: int = Field(description="Number of improving moves")
    num_worsening: int = Field(description="Number of worsening moves")


class CalibrateSACommand(BaseCommand):
    """Calibrate SA temperature by sampling the delta distribution."""

    config: "CalibrateSACommandConfig"

    def __init__(self, config: "CalibrateSACommandConfig") -> None:
        super().__init__(config)
        self.config: CalibrateSACommandConfig
        self._rng = random.Random(config.seed)

    async def run(self) -> None:
        """Execute SA calibration."""
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)

        logger.info(
            "SA calibration: target_acceptance=%.0f%%, num_samples=%d, "
            "max_steps=%d, T_guess=%.2f, max_ruin=%.2f, contiguous_p=%.1f",
            self.config.target_acceptance * 100,
            self.config.num_samples,
            self.config.max_steps,
            self.config.initial_temperature_guess,
            self.config.max_ruin_fraction,
            self.config.contiguous_ruin_probability,
        )

        # Create evaluator with proper context management
        async with self.config.evaluator.create_evaluator(
            output_dir=output_dir
        ) as evaluator:
            result = await self._run_calibration(evaluator)

        if result is not None:
            # Save using result handler
            ext = self.config.result_handler.handler_type
            result_handler = self.config.result_handler.create_result_handler()
            output_path = output_dir / f"calibration_results.{ext}"
            await result_handler.save(result, output_path)
            logger.info("Calibration results saved to %s", output_path)

    async def _run_calibration(self, evaluator) -> None | CalibrationResult:
        """Run calibration with evaluator."""
        problem = self.config.problem.create_problem(evaluator)

        if not isinstance(problem, CostConstrainedProblem):
            raise ValueError(
                "Calibration requires a cost-constrained problem "
                "(cost_max_performance or cost_min_cost)"
            )

        cost_model = problem.cost_model
        if not isinstance(cost_model, LinearCostModel):
            raise ValueError("Calibration currently only supports LinearCostModel")

        problem_config = self.config.problem
        if not isinstance(problem_config, MaxPerformanceProblemConfig):
            raise ValueError(
                "Calibration requires a cost_max_performance problem with cost_budget"
            )
        cost_budget = problem_config.cost_budget

        # Find a representative allocation at the cost bound
        num_layers = evaluator.num_layers
        allocation = cost_model.allocation_at_cost_bound(
            num_layers, cost_budget, from_above=True
        )

        if allocation is None:
            logger.error("Could not find allocation at cost bound.")
            return None

        logger.info(
            "Representative allocation: %s (cost=%.4f)",
            allocation,
            cost_model.allocation_cost(allocation),
        )

        # Sample delta distribution
        logger.info("Sampling %d ruin-and-recreate moves…", self.config.num_samples)
        deltas = await self._sample_deltas(problem, allocation)

        if not deltas:
            logger.error("No valid moves sampled (all were no-ops). Cannot calibrate.")
            return None

        positive = [d for d in deltas if d > 0]
        negative = [d for d in deltas if d < 0]

        logger.debug(
            "Delta distribution: %d total, %d improving (%.1f%%), %d worsening (%.1f%%)",
            len(deltas),
            len(positive),
            100 * len(positive) / len(deltas),
            len(negative),
            100 * len(negative) / len(deltas),
        )
        if positive:
            logger.debug(
                "  positive: min=%.4f, max=%.4f, avg=%.4f",
                min(positive),
                max(positive),
                sum(positive) / len(positive),
            )
        if negative:
            logger.debug(
                "  negative: min=%.4f, max=%.4f, avg=%.4f",
                min(negative),
                max(negative),
                sum(negative) / len(negative),
            )

        if not negative:
            logger.error("No worsening moves found! Cannot calibrate temperature.")
            return

        # Find temperature for target acceptance
        initial_temp = self._find_temperature(negative, self.config.target_acceptance)
        cooling_rate = self._compute_cooling_rate(
            initial_temp, self.config.min_temperature, self.config.max_steps
        )

        # Verify
        actual_acceptance = sum(math.exp(d / initial_temp) for d in negative) / len(
            negative
        )

        logger.info(
            "Calibration result: initial_temperature=%.4f, cooling_rate=%.6f, "
            "min_temperature=%s, max_steps=%d (acceptance at T0=%.1f%%)",
            initial_temp,
            cooling_rate,
            self.config.min_temperature,
            self.config.max_steps,
            actual_acceptance * 100,
        )

        # Convergence check
        ratio = initial_temp / self.config.initial_temperature_guess
        if ratio < 0.5 or ratio > 2.0:
            logger.warning(
                "Computed T_0 (%.4f) differs significantly from guess "
                "(%.4f). Consider re-running with "
                "initial_temperature_guess=%.2f for better accuracy.",
                initial_temp,
                self.config.initial_temperature_guess,
                initial_temp,
            )

        # Show table of acceptance ratios
        logger.debug("Temperature vs acceptance ratio:")
        for target in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            t = self._find_temperature(negative, target)
            logger.debug("  %.0f%% acceptance -> T = %.4f", target * 100, t)

        # Build structured result
        return CalibrationResult(
            initial_temperature=initial_temp,
            cooling_rate=cooling_rate,
            min_temperature=self.config.min_temperature,
            max_steps=self.config.max_steps,
            num_layers=num_layers,
            cost_budget=cost_budget,
            target_acceptance=self.config.target_acceptance,
            actual_acceptance=actual_acceptance,
            num_samples=self.config.num_samples,
            initial_temperature_guess=self.config.initial_temperature_guess,
            max_ruin_fraction=self.config.max_ruin_fraction,
            contiguous_ruin_probability=self.config.contiguous_ruin_probability,
            num_deltas=len(deltas),
            num_improving=len(positive),
            num_worsening=len(negative),
        )

    async def _sample_deltas(
        self,
        problem: CostConstrainedProblem,
        allocation: Allocation,
    ) -> list[float]:
        """Sample deltas using the ruin-and-recreate move operator.

        Uses the same move logic as AnnealingSearch:
        - Temperature-dependent perturbation size k (geometric distribution)
        - Contiguous or scattered ruin-and-recreate
        - Walks current state forward to sample from diverse states
        """
        num_layers = sum(allocation.values())
        current = random_assignment(self._rng, allocation, num_layers)

        placement = LayerPlacement(assignments=tuple(current))
        current_score = await problem.evaluate(placement)

        deltas = []
        for _ in range(self.config.num_samples):
            k = sample_perturbation_size(
                self._rng,
                num_layers,
                self.config.initial_temperature_guess,
                self.config.initial_temperature_guess,
                self.config.max_ruin_fraction,
            )
            proposed, _move_type = generate_move(
                self._rng,
                current,
                k,
                self.config.contiguous_ruin_probability,
            )

            # Skip no-ops (all ruined positions had same type)
            if proposed == current:
                continue

            placement = LayerPlacement(assignments=tuple(proposed))
            proposed_score = await problem.evaluate(placement)
            delta = proposed_score - current_score
            deltas.append(delta)

            # Walk state forward: accept improving always,
            # worsening with 50% to sample diverse states
            if delta > 0 or self._rng.random() < 0.5:
                current = proposed
                current_score = proposed_score

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
    problem: ProblemConfig = Field(description="Problem configuration")
    result_handler: ResultHandlerConfig = Field(
        description="Result handler configuration",
    )

    # Calibration parameters
    target_acceptance: float = Field(
        description="Target acceptance ratio for worsening moves (e.g., 0.8 = 80%)",
        gt=0,
        lt=1,
    )
    num_samples: int = Field(
        description="Number of ruin-and-recreate moves to sample",
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

    # Move operator parameters (must match SA search config)
    initial_temperature_guess: float = Field(
        description=(
            "Temperature guess for perturbation size sampling. "
            "Used to determine k distribution during calibration. "
            "Re-run if computed T_0 is far from this guess."
        ),
        gt=0,
    )
    max_ruin_fraction: float = Field(
        description="Max fraction of layers ruined at full temperature.",
        gt=0,
        le=1,
    )
    contiguous_ruin_probability: float = Field(
        description="Probability of contiguous block ruin vs random scattered ruin.",
        ge=0,
        le=1,
    )
