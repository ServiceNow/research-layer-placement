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
from pydantic import Field

from place_layers.command.base_command import BaseCommand, CommandError
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.core.types import LayerPlacement
from place_layers.cost_model import CostModelConfig
from place_layers.cost_model.base_cost_model import Allocation, BaseCostModel
from place_layers.cost_model.enumeration import enumerate_feasible_allocations
from place_layers.evaluator import EvaluatorConfig
from place_layers.evaluator.multi_evaluator_config import EvaluatorsConfig
from place_layers.problem import ProblemConfig
from place_layers.problem.layer_placement import MaxPerformanceProblemConfig
from place_layers.result_handler import ResultHandlerConfig
from place_layers.annealing.moves import random_assignment
from place_layers.surrogate.dataset import ExplorationDataset
from place_layers.surrogate.encoding import FeatureExpansionConfig, compute_feature_dim
from place_layers.surrogate.model_selection import (
    ExpansionSelectionResult,
    ridge_cv_rho,
    select_expansion_level,
)


logger = logging.getLogger(__name__)


# ============================================================================
# Result model
# ============================================================================


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

    logger.info("Loaded %d samples from %s", len(assignments), path)
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


def _compute_cost_uniform_weights(
    feasible_allocations: list[Allocation],
    cost_model: BaseCostModel,
) -> list[float]:
    """Compute sampling weights for cost-uniform allocation sampling.

    Each distinct cost level receives equal total weight, so the surrogate
    sees equal coverage across the cost spectrum regardless of how many
    allocations exist at each cost level.

    Without this, uniform-over-allocation sampling overweights cheap
    allocations (combinatorially more of them) and underweights the
    expensive region near the budget boundary — exactly where the search
    concentrates.
    """
    costs = [cost_model.allocation_cost(a) for a in feasible_allocations]
    # Group by cost level (round to avoid float precision issues)
    rounded = [round(c, 10) for c in costs]
    cost_counts: dict[float, int] = {}
    for rc in rounded:
        cost_counts[rc] = cost_counts.get(rc, 0) + 1
    return [1.0 / cost_counts[rc] for rc in rounded]


def _sample_batch(
    n: int,
    feasible_allocations: list[Allocation],
    num_layers: int,
    rng: random.Random,
    weights: list[float] | None = None,
) -> list[list[str]]:
    """Sample n random placements from feasible allocations.

    Each sample: pick a random allocation (weighted if *weights* provided),
    then generate a random assignment for it.

    Args:
        weights: Per-allocation sampling weights.  When provided, uses
            ``rng.choices`` (cost-uniform weighting); when *None*, uses
            ``rng.choices`` with equal weights.  Both paths consume the
            same number of RNG calls for deterministic resumption.
    """
    # Always use rng.choices for consistent RNG call pattern (1 call per sample)
    # regardless of whether custom weights are provided.
    indices = range(len(feasible_allocations))
    batch: list[list[str]] = []
    for _ in range(n):
        idx = rng.choices(indices, weights=weights, k=1)[0]
        allocation = feasible_allocations[idx]
        assignment = random_assignment(rng, allocation, num_layers)
        batch.append(assignment)
    return batch


def _fast_forward_rng(
    rng: random.Random,
    num_completed: int,
    feasible_allocations: list[Allocation],
    num_layers: int,
    weights: list[float] | None = None,
) -> None:
    """Advance the RNG to match the state after num_completed samples.

    This ensures deterministic resumption: generating the same sequence
    of random choices that were made for the already-completed samples.
    Replays the exact same calls as _sample_batch → random_assignment.
    """
    indices = range(len(feasible_allocations))
    for _ in range(num_completed):
        idx = rng.choices(indices, weights=weights, k=1)[0]
        allocation = feasible_allocations[idx]
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


def _log_expansion_result(
    result: ExpansionSelectionResult,
    n_samples: int,
    num_types: int,
    num_layers: int,
    previous_config: FeatureExpansionConfig | None = None,
) -> None:
    """Log a structured, indented expansion selection result block."""
    if previous_config is not None:
        header = f"Expansion re-selection ({n_samples} samples):"
    else:
        header = (
            f"Expansion selection "
            f"({n_samples} samples, {num_types} types, {num_layers} layers):"
        )

    lines = [header]

    # --- Phase 1 ---
    lines.append("  Phase 1 (range at order=2):")
    prev_rho: float | None = None
    for step in result.range_results:
        if step.config.max_order == 1:
            label = "unary baseline"
        else:
            label = f"pairwise range={step.config.pairwise_max_range}"
        entry = f"    {label}:{' ' * max(1, 20 - len(label))}{step.num_features} features, rho={step.rho:.4f}"
        if prev_rho is not None:
            delta = step.rho - prev_rho
            entry += f" ({delta:+.4f})"
        if step.stop_reason is not None:
            entry += f" <- {_stop_reason_label(step.stop_reason, n_samples)}"
        lines.append(entry)
        prev_rho = step.rho

    lines.append(f"  Selected range: {result.selected_range}")

    # --- Phase 2 ---
    if result.order_results:
        lines.append(
            f"  Phase 2 (order at range={result.selected_range}):"
        )
        # prev_rho carries over from phase 1's last step
        for step in result.order_results:
            label = f"order={step.config.max_order}"
            entry = f"    {label}:{' ' * max(1, 20 - len(label))}{step.num_features} features, rho={step.rho:.4f}"
            if prev_rho is not None:
                delta = step.rho - prev_rho
                entry += f" ({delta:+.4f})"
            if step.stop_reason is not None:
                entry += f" <- {_stop_reason_label(step.stop_reason, n_samples)}"
            lines.append(entry)
            prev_rho = step.rho
    else:
        # Phase 2 skipped — derive reason
        if result.phase1_stop_reason == "target_reached":
            skip_reason = "target reached in Phase 1"
        elif result.selected_order == 1:
            skip_reason = "unary only"
        else:
            skip_reason = "max_order < 3"
        lines.append(
            f"  Phase 2 (order at range={result.selected_range}): "
            f"skipped ({skip_reason})"
        )

    # --- Result line ---
    cfg = result.selected_config
    final_rho = prev_rho if prev_rho is not None else 0.0
    selected_features = compute_feature_dim(num_layers, num_types, cfg)
    result_line = (
        f"  Result: order={result.selected_order}, range={result.selected_range}, "
        f"{selected_features} features, rho={final_rho:.4f}"
    )

    # Upgrade annotation
    if previous_config is not None and previous_config != cfg:
        if previous_config.max_order == 1:
            old_desc = "unary"
        else:
            old_desc = (
                f"order={previous_config.max_order}, "
                f"range={previous_config.pairwise_max_range}"
            )
        result_line += f" [UPGRADED from {old_desc}]"

    lines.append(result_line)

    logger.info("\n".join(lines))


def _stop_reason_label(reason: str, n_samples: int) -> str:
    """Map a stop_reason string to a human-readable display label."""
    labels = {
        "target_reached": "target reached",
        "below_min_improvement": "below min_improvement",
        "feature_guard": f"feature guard (next would exceed {2 * n_samples})",
        "max_range": "max range",
        "max_order": "max order",
    }
    return labels.get(reason, reason)


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

        logger.info("Output directory: %s", output_dir)
        logger.info("Starting placement exploration...")

        async with self.config.evaluators.create_evaluators(
            output_dir
        ) as evaluator_instances:
            problem = self.config.problem.create_problem(evaluator_instances)

            cost_model = self.config.cost_model.create_cost_model()
            num_layers = evaluator_instances.num_layers
            

            # Determine cost budget for feasibility filtering
            cost_budget: float | None = None
            if isinstance(problem.config, MaxPerformanceProblemConfig):
                cost_budget = problem.config.cost_budget

            # Enumerate feasible allocations
            phase_order = cost_model.phase_order

            feasible_allocations = enumerate_feasible_allocations(
                cost_model, num_layers, cost_budget
            )
            logger.info(
                "Found %d feasible allocations for %d layers",
                len(feasible_allocations), num_layers,
            )
            logger.info(
                "Stopping criteria: target_rho=%g, "
                "plateau_patience=%d (min_improvement=%g), max_samples=%d",
                self.config.target_rank_correlation,
                self.config.plateau_patience,
                self.config.plateau_min_improvement,
                self.config.max_samples,
            )

            if not feasible_allocations:
                raise CommandError("No feasible allocations found within budget")

            # Cost-uniform sampling weights: each cost level gets equal total
            # weight so the surrogate sees balanced coverage across the cost
            # spectrum (avoids oversampling cheap allocations).
            sampling_weights = _compute_cost_uniform_weights(
                feasible_allocations, cost_model
            )

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
                    rng, len(all_assignments), feasible_allocations, num_layers,
                    weights=sampling_weights,
                )
                logger.info(
                    "Resumed from %d completed samples", len(all_assignments),
                )

            # --- Phase 1: Initial batch ---
            remaining_initial = max(
                0, self.config.min_samples - len(all_assignments)
            )
            if remaining_initial > 0:
                logger.info("Sampling initial batch of %d", remaining_initial)
                batch = _sample_batch(
                    remaining_initial, feasible_allocations, num_layers, rng,
                    weights=sampling_weights,
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

            # --- Phase 2: Initial expansion selection ---
            # Expansion selection is re-run in the adaptive loop (Phase 3) after
            # each batch. The feature-to-sample guard in select_expansion_level()
            # relaxes as more data arrives, potentially unlocking higher-order
            # features. Re-running is cheap (a few Ridge CV fits).
            feature_config: FeatureExpansionConfig | None = None
            if (
                self.config.expansion_max_order > 1
                and len(all_assignments) >= self.config.cv_folds
            ):
                expansion_result = select_expansion_level(
                    assignments=all_assignments,
                    scores=np.array(all_scores),
                    mixer_names=phase_order,
                    cv_folds=self.config.cv_folds,
                    target_rho=self.config.target_rank_correlation,
                    max_pairwise_range=self.config.expansion_max_range,
                    max_order=self.config.expansion_max_order,
                    seed=self.config.seed,
                )
                _log_expansion_result(
                    expansion_result,
                    n_samples=len(all_assignments),
                    num_types=len(phase_order),
                    num_layers=num_layers,
                )
                if expansion_result.selected_config.max_order > 1:
                    feature_config = expansion_result.selected_config
                else:
                    feature_config = None

            # --- Phase 2b: First quality check (Ridge CV) ---
            if len(all_assignments) >= self.config.cv_folds:
                rho = ridge_cv_rho(
                    assignments=all_assignments,
                    scores=np.array(all_scores),
                    mixer_names=phase_order,
                    feature_config=feature_config,
                    cv_folds=self.config.cv_folds,
                    seed=self.config.seed,
                )
                rho_history.append(rho)
                samples_at_check.append(len(all_assignments))
                logger.info(
                    "Ridge CV rho=%.4g (%d samples)",
                    rho,
                    len(all_assignments),
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
                        "Target rho (%g) reached: %.4g",
                        self.config.target_rank_correlation, rho,
                    )
                    break

                if _check_plateau(
                    rho_history,
                    self.config.plateau_patience,
                    self.config.plateau_min_improvement,
                ):
                    termination_reason = "plateau"
                    logger.info(
                        "Plateau detected after %d checks", len(rho_history),
                    )
                    break

                # Sample and evaluate next batch
                batch_size = min(
                    self.config.batch_size,
                    self.config.max_samples - len(all_assignments),
                )
                logger.info(
                    "Sampling batch of %d (total: %d)",
                    batch_size, len(all_assignments),
                )
                batch = _sample_batch(
                    batch_size, feasible_allocations, num_layers, rng,
                    weights=sampling_weights,
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

                # Re-run expansion selection with larger dataset.
                # The feature-to-sample guard relaxes as data grows,
                # potentially unlocking higher-order features.
                if (
                    self.config.expansion_max_order > 1
                    and len(all_assignments) >= self.config.cv_folds
                ):
                    expansion_result = select_expansion_level(
                        assignments=all_assignments,
                        scores=np.array(all_scores),
                        mixer_names=phase_order,
                        cv_folds=self.config.cv_folds,
                        target_rho=self.config.target_rank_correlation,
                        max_pairwise_range=self.config.expansion_max_range,
                        max_order=self.config.expansion_max_order,
                        seed=self.config.seed,
                    )
                    new_config = (
                        expansion_result.selected_config
                        if expansion_result.selected_config.max_order > 1
                        else None
                    )
                    if new_config != feature_config:
                        old_feature_config = feature_config
                        feature_config = new_config
                        _log_expansion_result(
                            expansion_result,
                            n_samples=len(all_assignments),
                            num_types=len(phase_order),
                            num_layers=num_layers,
                            previous_config=old_feature_config,
                        )

                # Re-check quality via Ridge CV
                rho = ridge_cv_rho(
                    assignments=all_assignments,
                    scores=np.array(all_scores),
                    mixer_names=phase_order,
                    feature_config=feature_config,
                    cv_folds=self.config.cv_folds,
                    seed=self.config.seed,
                )
                rho_delta = rho - rho_history[-1] if rho_history else rho
                # NOTE: rho_history may reflect different feature configs at
                # different checkpoints if expansion was upgraded mid-loop.
                # A jump from e.g. 0.93 (order-2) to 0.96 (order-3) is real
                # progress and the plateau detector correctly sees it as such.
                rho_history.append(rho)
                samples_at_check.append(len(all_assignments))
                if feature_config is not None:
                    logger.info(
                        "Ridge CV rho=%.4g (%+.4g) at %d samples "
                        "[order=%d, range=%d]",
                        rho,
                        rho_delta,
                        len(all_assignments),
                        feature_config.max_order,
                        feature_config.pairwise_max_range,
                    )
                else:
                    logger.info(
                        "Ridge CV rho=%.4g (%+.4g) at %d samples [unary]",
                        rho,
                        rho_delta,
                        len(all_assignments),
                    )

        # Build result dataset
        assert rho_history, (
            "Quality check was never run — need at least cv_folds samples"
        )
        dataset = ExplorationDataset(
            assignments=all_assignments,
            scores=all_scores,
            allocations=all_allocations,
            num_layers=num_layers,
            mixer_names=phase_order,
            num_samples=len(all_assignments),
            termination_reason=termination_reason,
            feature_config=feature_config,
            rho_history=rho_history,
            samples_at_check=samples_at_check,
            final_rho=rho_history[-1] if rho_history else 0.0,
        )

        # Save using result handler
        result_handler = self.config.result_handler.create_result_handler()
        await result_handler.save(dataset, output_dir / "exploration_results.json")

        logger.info(
            "Exploration complete: %d/%d samples, final rho=%.4g, stopped=%s",
            dataset.num_samples,
            self.config.max_samples,
            dataset.final_rho,
            dataset.termination_reason,
        )


# ============================================================================
# Config
# ============================================================================


class ExplorePlacementsCommandConfig(BaseCommandConfig):
    """Configuration for explore_placements command.

    All defaults are in the YAML config file (conf/command/explore_placements.yaml).
    """

    _impl_class: ClassVar[type[BaseCommand]] = ExplorePlacementsCommand

    command: Literal["explore_placements"] = "explore_placements"

    evaluators: EvaluatorsConfig = Field(
        description="Evaluator configuration (single or multiple with aggregation)"
    )
    problem: ProblemConfig = Field(description="Problem configuration")
    cost_model: CostModelConfig = Field(
        description="Cost model for allocation enumeration and sampling weights.",
    )
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
        description=(
            "Stop when Ridge CV Spearman rho reaches this value. "
            "Also used as the target for expansion level selection."
        ),
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

    seed: int = Field(description="Random seed for reproducibility.")

    # Feature expansion selection
    expansion_max_order: int = Field(
        description=(
            "Max n-body order to test for expansion selection. "
            "1 = skip expansion selection (unary only, backward compatible)."
        ),
        ge=1,
    )
    expansion_max_range: int = Field(
        description=(
            "Ceiling for pairwise skip distance tested during expansion selection. "
            "The search stops earlier if marginal improvement vanishes or the "
            "feature-to-sample ratio becomes unfavorable."
        ),
        ge=1,
    )
