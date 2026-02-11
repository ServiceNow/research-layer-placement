"""Explore placements command: build dataset for surrogate model training.

Uses adaptive sampling to evaluate random placements until surrogate quality
(Spearman rank correlation) reaches a target or plateaus. The dataset can
then be used by the surrogate_search command.

Resumability
------------
Each evaluated (assignment, score, allocation) is written incrementally to a
JSONL file. On restart, this file is loaded to recover progress. The RNG state
is deterministic: with the same seed and sample count, the same next batch is
generated, so exploration continues from the exact same point.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from hydra.core.hydra_config import HydraConfig
from pydantic import BaseModel, Field

from place_layers.command.base_command import BaseCommand, CommandError
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.core.types import LayerPlacement
from place_layers.cost_model.base_cost_model import Allocation, BaseCostModel
from place_layers.evaluator import EvaluatorConfig
from place_layers.problem import ProblemConfig
from place_layers.problem.layer_placement import CostConstrainedProblem
from place_layers.result_handler import ResultHandlerConfig
from place_layers.annealing.moves import random_assignment
from place_layers.surrogate.forest_surrogate import ForestSurrogate


logger = logging.getLogger(__name__)


# ============================================================================
# Result model
# ============================================================================


class ExplorationDataset(BaseModel):
    """Serializable dataset of evaluated placements."""

    model_config = {"frozen": True}

    # Data
    assignments: list[tuple[str, ...]]
    scores: list[float]
    allocations: list[dict[str, int]]

    # Metadata
    num_layers: int
    mixer_names: list[str]
    num_samples: int
    termination_reason: Literal["target_reached", "plateau", "max_samples"]

    # Learning curve
    rho_history: list[float]
    samples_at_check: list[int]
    final_rho: float

    @classmethod
    def load(cls, path: Path) -> "ExplorationDataset":
        """Load dataset from JSON file."""
        return cls.model_validate_json(path.read_text())


# ============================================================================
# Helpers
# ============================================================================


def _load_progress(
    path: Path,
) -> tuple[list[tuple[str, ...]], list[float], list[dict[str, int]]]:
    """Load incremental JSONL progress file.

    Returns:
        (assignments, scores, allocations) — each a list of the same length.
    """
    assignments: list[tuple[str, ...]] = []
    scores: list[float] = []
    allocations: list[dict[str, int]] = []

    if not path.exists():
        return assignments, scores, allocations

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            assignments.append(tuple(record["assignment"]))
            scores.append(record["score"])
            allocations.append(record["allocation"])

    logger.info(f"Loaded {len(assignments)} samples from {path}")
    return assignments, scores, allocations


def _append_to_jsonl(
    path: Path,
    assignment: tuple[str, ...] | list[str],
    score: float,
    allocation: dict[str, int],
) -> None:
    """Append one evaluated record to the progress JSONL file."""
    record = {
        "assignment": list(assignment),
        "score": score,
        "allocation": allocation,
    }
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _get_allocation(assignment: list[str] | tuple[str, ...]) -> dict[str, int]:
    """Compute allocation counts from an assignment."""
    alloc: dict[str, int] = {}
    for mixer in assignment:
        alloc[mixer] = alloc.get(mixer, 0) + 1
    return alloc


def _enumerate_feasible_allocations(
    cost_model: BaseCostModel,
    num_layers: int,
    cost_budget: float | None,
) -> list[Allocation]:
    """Collect all allocations within budget (or all if no budget).

    Args:
        cost_model: The cost model to query.
        num_layers: Number of layers.
        cost_budget: If set, only include allocations with cost <= budget.

    Returns:
        List of feasible allocations.
    """
    feasible: list[Allocation] = []
    for alloc_cost, allocation in cost_model.allocations_by_cost(
        num_layers, direction="increasing"
    ):
        if cost_budget is not None and alloc_cost > cost_budget:
            break  # All subsequent allocations are more expensive
        feasible.append(allocation)
    return feasible


def _sample_batch(
    n: int,
    feasible_allocations: list[Allocation],
    num_layers: int,
    rng: random.Random,
) -> list[list[str]]:
    """Sample n random placements from feasible allocations.

    Each sample: pick a random allocation, then generate a random assignment.
    """
    batch: list[list[str]] = []
    for _ in range(n):
        allocation = rng.choice(feasible_allocations)
        assignment = random_assignment(rng, allocation, num_layers)
        batch.append(assignment)
    return batch


def _fast_forward_rng(
    rng: random.Random,
    num_completed: int,
    feasible_allocations: list[Allocation],
    num_layers: int,
) -> None:
    """Advance the RNG to match the state after num_completed samples.

    This ensures deterministic resumption: generating the same sequence
    of random choices that were made for the already-completed samples.
    Replays the exact same calls as _sample_batch → random_assignment.
    """
    for _ in range(num_completed):
        allocation = rng.choice(feasible_allocations)
        # random_assignment builds a list and shuffles — consume same RNG calls
        random_assignment(rng, allocation, num_layers)


def _check_plateau(
    rho_history: list[float],
    patience: int,
    min_improvement: float,
) -> bool:
    """Return True if last `patience` improvements are all < min_improvement."""
    if len(rho_history) < patience + 1:
        return False
    recent = rho_history[-(patience + 1) :]
    for i in range(1, len(recent)):
        if recent[i] - recent[i - 1] >= min_improvement:
            return False
    return True


# ============================================================================
# Command
# ============================================================================


class ExplorePlacementsCommand(BaseCommand):
    """Build a dataset of evaluated placements for surrogate training."""

    config: "ExplorePlacementsCommandConfig"

    async def run(self) -> None:
        """Execute adaptive exploration loop."""
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)

        logger.info(f"Output directory: {output_dir}")
        logger.info("Starting placement exploration...")

        async with self.config.evaluator.create_evaluator(output_dir) as evaluator:
            problem = self.config.problem.create_problem(evaluator)

            if not isinstance(problem, CostConstrainedProblem):
                raise CommandError(
                    "explore_placements requires a CostConstrainedProblem "
                    f"(got {type(problem).__name__})"
                )

            cost_model = problem.cost_model
            num_layers = evaluator.num_layers

            # Determine cost budget for feasibility filtering
            from place_layers.problem.layer_placement import MaxPerformanceProblemConfig

            cost_budget: float | None = None
            if isinstance(problem.config, MaxPerformanceProblemConfig):
                cost_budget = problem.config.cost_budget

            # Enumerate feasible allocations
            phase_order = cost_model.phase_order

            feasible_allocations = _enumerate_feasible_allocations(
                cost_model, num_layers, cost_budget
            )
            logger.info(
                f"Found {len(feasible_allocations)} feasible allocations "
                f"for {num_layers} layers"
            )

            if not feasible_allocations:
                raise CommandError("No feasible allocations found within budget")

            # Load existing progress
            progress_path = output_dir / "exploration_progress.jsonl"
            all_assignments, all_scores, all_allocations = _load_progress(
                progress_path
            )

            rho_history: list[float] = []
            samples_at_check: list[int] = []

            # Create RNG and fast-forward past completed samples
            rng = random.Random(self.config.seed)
            if all_assignments:
                _fast_forward_rng(
                    rng, len(all_assignments), feasible_allocations, num_layers
                )
                logger.info(
                    f"Resumed from {len(all_assignments)} completed samples"
                )

            # --- Phase 1: Initial batch ---
            remaining_initial = max(
                0, self.config.min_samples - len(all_assignments)
            )
            if remaining_initial > 0:
                logger.info(f"Sampling initial batch of {remaining_initial}")
                batch = _sample_batch(
                    remaining_initial, feasible_allocations, num_layers, rng
                )
                for assignment in batch:
                    placement = LayerPlacement(assignments=tuple(assignment))
                    score = await problem.evaluate(placement)
                    assignment_tuple = tuple(assignment)
                    allocation = _get_allocation(assignment)
                    all_assignments.append(assignment_tuple)
                    all_scores.append(score)
                    all_allocations.append(allocation)
                    _append_to_jsonl(
                        progress_path, assignment_tuple, score, allocation
                    )

            # Create surrogate for quality checks
            surrogate = ForestSurrogate(
                n_estimators=self.config.rf_n_estimators,
                max_depth=None,
                seed=self.config.seed,
                mixer_names=phase_order,
            )

            # --- Phase 2: First quality check ---
            if len(all_assignments) >= self.config.cv_folds:
                rho = surrogate.cross_validate(
                    all_assignments, np.array(all_scores), self.config.cv_folds
                )
                rho_history.append(rho)
                samples_at_check.append(len(all_assignments))
                logger.info(
                    f"Initial quality check: rho={rho:.4f} "
                    f"({len(all_assignments)} samples)"
                )
            else:
                rho = 0.0

            # --- Phase 3: Adaptive loop ---
            termination_reason: Literal[
                "target_reached", "plateau", "max_samples"
            ] = "max_samples"

            while len(all_assignments) < self.config.max_samples:
                # Check stopping criteria
                if rho >= self.config.target_rank_correlation:
                    termination_reason = "target_reached"
                    logger.info(
                        f"Target rho ({self.config.target_rank_correlation}) "
                        f"reached: {rho:.4f}"
                    )
                    break

                if _check_plateau(
                    rho_history,
                    self.config.plateau_patience,
                    self.config.plateau_min_improvement,
                ):
                    termination_reason = "plateau"
                    logger.info(
                        f"Plateau detected after {len(rho_history)} checks"
                    )
                    break

                # Sample and evaluate next batch
                batch_size = min(
                    self.config.batch_size,
                    self.config.max_samples - len(all_assignments),
                )
                logger.info(
                    f"Sampling batch of {batch_size} "
                    f"(total: {len(all_assignments)})"
                )
                batch = _sample_batch(
                    batch_size, feasible_allocations, num_layers, rng
                )
                for assignment in batch:
                    placement = LayerPlacement(assignments=tuple(assignment))
                    score = await problem.evaluate(placement)
                    assignment_tuple = tuple(assignment)
                    allocation = _get_allocation(assignment)
                    all_assignments.append(assignment_tuple)
                    all_scores.append(score)
                    all_allocations.append(allocation)
                    _append_to_jsonl(
                        progress_path, assignment_tuple, score, allocation
                    )

                # Re-check quality
                rho = surrogate.cross_validate(
                    all_assignments, np.array(all_scores), self.config.cv_folds
                )
                rho_history.append(rho)
                samples_at_check.append(len(all_assignments))
                logger.info(
                    f"Quality check: rho={rho:.4f} "
                    f"({len(all_assignments)} samples)"
                )

        # Build result dataset
        dataset = ExplorationDataset(
            assignments=all_assignments,
            scores=all_scores,
            allocations=all_allocations,
            num_layers=num_layers,
            mixer_names=phase_order,
            num_samples=len(all_assignments),
            termination_reason=termination_reason,
            rho_history=rho_history,
            samples_at_check=samples_at_check,
            final_rho=rho_history[-1] if rho_history else 0.0,
        )

        # Save using result handler
        result_handler = self.config.result_handler.create_result_handler()
        await result_handler.save(dataset, output_dir / "exploration_results.json")

        logger.info(
            f"Exploration complete: {dataset.num_samples} samples, "
            f"final rho={dataset.final_rho:.4f}, "
            f"reason={dataset.termination_reason}"
        )


# ============================================================================
# Config
# ============================================================================


class ExplorePlacementsCommandConfig(BaseCommandConfig):
    """Configuration for explore_placements command.

    All defaults are in the YAML config file (conf/command/explore_placements.yaml).
    """

    _command_class: ClassVar[type[BaseCommand]] = ExplorePlacementsCommand

    command: Literal["explore_placements"] = "explore_placements"

    evaluator: EvaluatorConfig = Field(description="Evaluator configuration")
    problem: ProblemConfig = Field(description="Problem configuration")
    result_handler: ResultHandlerConfig = Field(
        description="Result handler configuration"
    )

    # Adaptive sampling
    min_samples: int = Field(
        description="Initial batch size before first surrogate quality check.",
        gt=0,
    )
    batch_size: int = Field(
        description="Samples per subsequent batch.",
        gt=0,
    )
    max_samples: int = Field(
        description="Hard cap on total evaluations.",
        gt=0,
    )

    # Stopping criteria
    target_rank_correlation: float = Field(
        description="Stop when Spearman rho reaches this value.",
        gt=0,
        le=1,
    )
    plateau_patience: int = Field(
        description="Stop if rho doesn't improve for this many batches.",
        gt=0,
    )
    plateau_min_improvement: float = Field(
        description="Minimum rho improvement to count as progress.",
        ge=0,
    )
    cv_folds: int = Field(
        description="Number of cross-validation folds for quality estimation.",
        gt=1,
    )

    # Surrogate config for quality checks
    rf_n_estimators: int = Field(
        description="Number of trees in the random forest surrogate.",
        gt=0,
    )

    seed: int = Field(description="Random seed for reproducibility.")
