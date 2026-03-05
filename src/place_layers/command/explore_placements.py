"""Explore placements command: build dataset for surrogate model training.

Uses adaptive sampling to evaluate random placements until surrogate quality
reaches a target or plateaus. The dataset can then be used by the
surrogate_search command.

Resumability
------------
Each evaluated (assignment, score) is written incrementally to a
JSONL file. On restart, this file is loaded to recover progress. The RNG state
is deterministic: with the same seed and sample count, the same next batch is
generated, so exploration continues from the exact same point.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, assert_never

import numpy as np
from hydra.core.hydra_config import HydraConfig
from pydantic import Field, model_validator

from place_layers.command.base_command import BaseCommand, CommandError
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.core.types import LayerPlacement
from place_layers.cost_model import CostModelConfig
from place_layers.landscape.config import objective_result_path
from place_layers.cost_model.base_cost_model import Allocation, BaseCostModel
from place_layers.cost_model.enumeration import enumerate_feasible_allocations
from place_layers.backend import BackendConfig
from place_layers.evaluator.evaluators import EvaluatorsConfig
from place_layers.problem import ProblemConfig
from place_layers.problem.layer_placement import MaxPerformanceProblemConfig
from place_layers.result_handler import ResultHandlerConfig
from place_layers.annealing.moves import random_assignment
from place_layers.surrogate.dataset import ExplorationDataset
from place_layers.surrogate.evaluation_log import (
    EVALUATION_RECORDS_FILENAME,
    EvaluationLogMeta,
    EvaluationRecord,
    append_record,
    load_records,
    write_meta,
)
from place_layers.surrogate.encoding import FeatureExpansionConfig, compute_feature_dim
from place_layers.surrogate.config import (
    BayesianRidgeSurrogateConfig,
    RidgeSurrogateConfig,
    SurrogateConfig,
)
from place_layers.surrogate.model_selection import (
    ExpansionSelectionResult,
    ridge_cv_rho,
    select_expansion_bayes_evidence,
    select_expansion_ridge_cv,
)

if TYPE_CHECKING:
    from place_layers.surrogate.bayesian_ridge import BayesianRidgeSurrogate


logger = logging.getLogger(__name__)


# ============================================================================
# Result model
# ============================================================================


# ============================================================================
# Helpers
# ============================================================================


def _load_progress(
    path: Path,
) -> tuple[list[tuple[str, ...]], list[float]]:
    """Load incremental JSONL progress file.

    Returns:
        (assignments, scores) — each a list of the same length.
    """
    if not path.exists():
        return [], []

    records = load_records(path)
    if records:
        logger.info("Loaded %d samples from %s", len(records), path)
    return (
        [r.assignment for r in records],
        [r.score for r in records],
    )


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
    quality_history: list[float],
    patience: int,
    min_improvement: float,
) -> bool:
    """Return True if last `patience` improvements are all < min_improvement."""
    if len(quality_history) < patience + 1:
        return False
    recent = quality_history[-(patience + 1) :]
    for i in range(1, len(recent)):
        if recent[i] - recent[i - 1] >= min_improvement:
            return False
    return True


def _log_expansion_result(
    result: ExpansionSelectionResult,
    n_samples: int,
    num_types: int,
    num_layers: int,
    surrogate_config: SurrogateConfig,
    previous_config: FeatureExpansionConfig | None = None,
) -> None:
    """Log a structured, indented expansion selection result block."""
    match surrogate_config:
        case BayesianRidgeSurrogateConfig():
            method = "Bayesian evidence"
            score_label = "evidence"
        case RidgeSurrogateConfig():
            method = "RidgeCV Spearman"
            score_label = "rho"
        case other:
            assert_never(other)

    if previous_config is not None:
        header = f"Expansion re-selection via {method} ({n_samples} samples):"
    else:
        header = (
            f"Expansion selection via {method} "
            f"({n_samples} samples, {num_types} types, {num_layers} layers):"
        )

    lines = [header]

    # --- Phase 1 ---
    lines.append("  Phase 1 (range at order=2):")
    prev_score: float | None = None
    for step in result.range_results:
        if step.config.max_order == 1:
            label = "unary baseline"
        else:
            label = f"pairwise range={step.config.pairwise_max_range}"
        entry = f"    {label}:{' ' * max(1, 20 - len(label))}{step.num_features} features, {score_label}={step.score:.4f}"
        if prev_score is not None:
            delta = step.score - prev_score
            entry += f" ({delta:+.4f})"
        if step.stop_reason is not None:
            entry += f" <- {_stop_reason_label(step.stop_reason, n_samples)}"
        lines.append(entry)
        prev_score = step.score

    lines.append(f"  Selected range: {result.selected_range}")

    # --- Phase 2 ---
    if result.order_results:
        lines.append(f"  Phase 2 (order at range={result.selected_range}):")
        # prev_score carries over from phase 1's last step
        for step in result.order_results:
            label = f"order={step.config.max_order}"
            entry = f"    {label}:{' ' * max(1, 20 - len(label))}{step.num_features} features, {score_label}={step.score:.4f}"
            if prev_score is not None:
                delta = step.score - prev_score
                entry += f" ({delta:+.4f})"
            if step.stop_reason is not None:
                entry += f" <- {_stop_reason_label(step.stop_reason, n_samples)}"
            lines.append(entry)
            prev_score = step.score
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
    final_score = prev_score if prev_score is not None else 0.0
    selected_features = compute_feature_dim(num_layers, num_types, cfg)
    result_line = (
        f"  Result: order={result.selected_order}, range={result.selected_range}, "
        f"{selected_features} features, {score_label}={final_score:.4f}"
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
        "exhaustive_argmax": "exhaustive argmax",
    }
    return labels.get(reason, reason)


# ============================================================================
# Command
# ============================================================================


class ExplorePlacementsCommand(BaseCommand):
    """Build a dataset of evaluated placements for surrogate training."""

    config: "ExplorePlacementsCommandConfig"

    # Stashed surrogate for Bayesian batch predictive scoring.
    # None means the next quality check is "stash only" (no score returned).
    _prev_surrogate: BayesianRidgeSurrogate | None = None

    async def run(self) -> None:
        """Execute adaptive exploration loop."""
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)

        logger.info("Output directory: %s", output_dir)
        logger.info("Starting placement exploration...")

        from contextlib import nullcontext

        backend_cm = (
            self.config.backend.create_backend()
            if self.config.backend
            else nullcontext()
        )
        async with backend_cm as backend:
            async with self.config.evaluators.create_evaluators(
                output_dir, backend=backend
            ) as evaluator_instances:
                cost_model = self.config.cost_model.create_cost_model()
                num_layers = evaluator_instances.num_layers

                for obj_name in evaluator_instances.objective_names:
                    await self._explore_objective(
                        evaluator_instances=evaluator_instances,
                        obj_name=obj_name,
                        cost_model=cost_model,
                        num_layers=num_layers,
                        output_dir=output_dir,
                    )

    async def _explore_objective(
        self,
        evaluator_instances,
        obj_name: str,
        cost_model: BaseCostModel,
        num_layers: int,
        output_dir: Path,
    ) -> None:
        """Run the adaptive exploration loop for a single objective."""
        logger.info("Exploring objective: %s", obj_name)
        logger.info("Surrogate: %s", self.config.surrogate.surrogate_type)

        problem = self.config.problem.create_problem(
            evaluator_instances, objective_name=obj_name
        )

        # Determine cost budget and floor for feasibility filtering
        cost_budget: float | None = None
        cost_floor: float | None = None
        if isinstance(problem.config, MaxPerformanceProblemConfig):
            cost_budget = problem.config.cost_budget
            cost_floor = problem.config.cost_floor

        # Enumerate feasible allocations
        phase_order = cost_model.phase_order

        feasible_allocations = enumerate_feasible_allocations(
            cost_model, num_layers, cost_budget, cost_floor=cost_floor
        )
        logger.info(
            "Found %d feasible allocations for %d layers",
            len(feasible_allocations),
            num_layers,
        )
        match self.config.surrogate:
            case RidgeSurrogateConfig() as cfg:
                logger.info(
                    "Stopping criteria: target_rho=%g, "
                    "plateau_patience=%d (min_improvement=%g), max_samples=%d",
                    cfg.target_rank_correlation,
                    self.config.plateau_patience,
                    self.config.plateau_min_improvement,
                    self.config.max_samples,
                )
            case BayesianRidgeSurrogateConfig():
                logger.info(
                    "Stopping criteria: plateau on batch predictive score, "
                    "plateau_patience=%d (min_improvement=%g), max_samples=%d",
                    self.config.plateau_patience,
                    self.config.plateau_min_improvement,
                    self.config.max_samples,
                )
            case other:
                assert_never(other)

        if not feasible_allocations:
            raise CommandError("No feasible allocations found within budget")

        # Cost-uniform sampling weights: each cost level gets equal total
        # weight so the surrogate sees balanced coverage across the cost
        # spectrum (avoids oversampling cheap allocations).
        sampling_weights = _compute_cost_uniform_weights(
            feasible_allocations, cost_model
        )

        # Load existing progress (per-objective JSONL)
        obj_dir = output_dir / obj_name
        obj_dir.mkdir(parents=True, exist_ok=True)
        progress_path = obj_dir / "progress.jsonl"
        all_assignments, all_scores = _load_progress(progress_path)

        # Capture local count before merging supplemental data
        num_local = len(all_assignments)
        n_supplemental = 0

        if self.config.supplemental_data_dirs:
            from place_layers.surrogate.evaluation_log import (
                discover_evaluation_records,
                load_pooled_data,
            )

            supplemental_files = discover_evaluation_records(
                self.config.supplemental_data_dirs, obj_name
            )
            if supplemental_files:
                pool = load_pooled_data(supplemental_files, deduplicate=True)
                existing = {(a, s) for a, s in zip(all_assignments, all_scores)}
                n_before = len(all_assignments)
                for a, s in zip(pool.assignments, pool.scores):
                    if (a, s) not in existing:
                        all_assignments.append(a)
                        all_scores.append(s)
                        existing.add((a, s))
                n_added = len(all_assignments) - n_before
                n_supplemental = n_added
                logger.info(
                    "Supplemental data: loaded %d records from %d file(s), "
                    "merged %d new (skipped %d duplicates) for objective '%s'",
                    pool.num_samples,
                    len(supplemental_files),
                    n_added,
                    pool.num_samples - n_added,
                    obj_name,
                )
            else:
                logger.info(
                    "Supplemental data: no files found for objective '%s' in %d director%s",
                    obj_name,
                    len(self.config.supplemental_data_dirs),
                    "y" if len(self.config.supplemental_data_dirs) == 1 else "ies",
                )

        quality_history: list[float] = []
        samples_at_check: list[int] = []

        # Create RNG and fast-forward past completed samples
        rng = random.Random(self.config.seed)
        if num_local > 0:
            _fast_forward_rng(
                rng,
                num_local,
                feasible_allocations,
                num_layers,
                weights=sampling_weights,
            )
            if len(all_assignments) > num_local:
                logger.info(
                    "Resumed from %d local samples (%d total with supplemental)",
                    num_local,
                    len(all_assignments),
                )
            else:
                logger.info("Resumed from %d completed samples", num_local)

        # Write sidecar metadata for the progress JSONL so it can be loaded
        # standalone by load_pooled_data (without a companion .json file).
        from datetime import datetime, timezone

        progress_meta = EvaluationLogMeta(
            version=1,
            objective_name=obj_name,
            num_layers=num_layers,
            mixer_names=list(phase_order),
            source_command="explore_placements",
            source_detail="exploration_progress",
            created_at=datetime.now(timezone.utc).isoformat(),
            num_records=None,
        )
        write_meta(progress_path, progress_meta)

        # Open snowball file for incremental writes (supplemental + local).
        # Written alongside progress file; discoverable for chaining.
        snowball_path = obj_dir / EVALUATION_RECORDS_FILENAME
        snowball_meta = EvaluationLogMeta(
            version=1,
            objective_name=obj_name,
            num_layers=num_layers,
            mixer_names=list(phase_order),
            source_command="explore_placements",
            source_detail="merged_snowball",
            created_at=datetime.now(timezone.utc).isoformat(),
            num_records=None,
        )
        write_meta(snowball_path, snowball_meta)

        # Open progress file for incremental writes
        progress_file = progress_path.open("a")
        snowball_file = snowball_path.open("w")
        try:
            # Seed snowball with all records loaded so far (local progress + supplemental)
            for a, s in zip(all_assignments, all_scores):
                append_record(
                    snowball_file,
                    EvaluationRecord(
                        assignment=a,
                        score=s,
                    ),
                )
            # --- Phase 1: Initial batch ---
            remaining_initial = max(0, self.config.min_samples - len(all_assignments))
            if remaining_initial > 0:
                logger.info("Sampling initial batch of %d", remaining_initial)
                batch = _sample_batch(
                    remaining_initial,
                    feasible_allocations,
                    num_layers,
                    rng,
                    weights=sampling_weights,
                )
                for assignment in batch:
                    placement = LayerPlacement(assignments=tuple(assignment))
                    score = await problem.evaluate(placement)
                    assignment_tuple = tuple(assignment)
                    all_assignments.append(assignment_tuple)
                    all_scores.append(score)
                    record = EvaluationRecord(
                        assignment=assignment_tuple,
                        score=score,
                    )
                    append_record(progress_file, record)
                    append_record(snowball_file, record)

            # --- Phase 2: Initial expansion selection ---
            # Expansion selection is re-run in the adaptive loop (Phase 3) after
            # each batch. The feature-to-sample guard in _select_expansion()
            # relaxes as more data arrives, potentially unlocking higher-order
            # features. Re-running is cheap (a few surrogate fits).
            feature_config = FeatureExpansionConfig.unary()
            last_expansion_result: ExpansionSelectionResult | None = None
            min_for_check = self._min_samples_for_quality_check()
            if (
                self.config.expansion_max_order > 1
                and len(all_assignments) >= min_for_check
            ):
                expansion_result = self._select_expansion(
                    all_assignments,
                    np.array(all_scores),
                    phase_order,
                )
                last_expansion_result = expansion_result
                _log_expansion_result(
                    expansion_result,
                    n_samples=len(all_assignments),
                    num_types=len(phase_order),
                    num_layers=num_layers,
                    surrogate_config=self.config.surrogate,
                )
                feature_config = expansion_result.selected_config

            # --- Phase 2b: First quality check ---
            # Initialize Bayesian stashed-surrogate tracking.
            self._prev_surrogate = None

            if len(all_assignments) >= min_for_check:
                quality_or_none = self._check_quality(
                    all_assignments,
                    np.array(all_scores),
                    phase_order,
                    feature_config,
                )
                if quality_or_none is not None:
                    quality = quality_or_none
                    quality_history.append(quality)
                    samples_at_check.append(len(all_assignments))
                    logger.info(
                        "Quality=%.4g (%d samples)",
                        quality,
                        len(all_assignments),
                    )
                else:
                    # Bayesian stash-only: no quality value yet
                    logger.info(
                        "Bayesian: initialized posterior checkpoint (%d samples)",
                        len(all_assignments),
                    )

            # --- Phase 3: Adaptive loop ---
            termination_reason: Literal["target_reached", "plateau", "max_samples"] = (
                "max_samples"
            )

            while len(all_assignments) < self.config.max_samples:
                # Check stopping criteria
                if quality_history and self._check_target_reached(quality_history[-1]):
                    termination_reason = "target_reached"
                    logger.info(
                        "Target quality reached: %.4g",
                        quality_history[-1],
                    )
                    break

                if _check_plateau(
                    quality_history,
                    self.config.plateau_patience,
                    self.config.plateau_min_improvement,
                ):
                    termination_reason = "plateau"
                    logger.info(
                        "Plateau detected after %d checks",
                        len(quality_history),
                    )
                    break

                # Sample and evaluate next batch
                batch_size = min(
                    self.config.batch_size,
                    self.config.max_samples - len(all_assignments),
                )
                logger.info(
                    "Sampling batch of %d (total: %d)",
                    batch_size,
                    len(all_assignments),
                )
                batch = _sample_batch(
                    batch_size,
                    feasible_allocations,
                    num_layers,
                    rng,
                    weights=sampling_weights,
                )
                for assignment in batch:
                    placement = LayerPlacement(assignments=tuple(assignment))
                    score = await problem.evaluate(placement)
                    assignment_tuple = tuple(assignment)
                    all_assignments.append(assignment_tuple)
                    all_scores.append(score)
                    record = EvaluationRecord(
                        assignment=assignment_tuple,
                        score=score,
                    )
                    append_record(progress_file, record)
                    append_record(snowball_file, record)

                # Re-run expansion selection with larger dataset.
                # The feature-to-sample guard relaxes as data grows,
                # potentially unlocking higher-order features.
                if (
                    self.config.expansion_max_order > 1
                    and len(all_assignments) >= min_for_check
                ):
                    expansion_result = self._select_expansion(
                        all_assignments,
                        np.array(all_scores),
                        phase_order,
                    )
                    last_expansion_result = expansion_result
                    new_config = expansion_result.selected_config
                    if new_config != feature_config:
                        old_feature_config = feature_config
                        feature_config = new_config
                        # Expansion changed: reset stashed surrogate so next
                        # quality check reinitializes in the new feature space.
                        self._prev_surrogate = None
                        _log_expansion_result(
                            expansion_result,
                            n_samples=len(all_assignments),
                            num_types=len(phase_order),
                            num_layers=num_layers,
                            surrogate_config=self.config.surrogate,
                            previous_config=old_feature_config,
                        )

                # Re-check quality
                quality_or_none = self._check_quality(
                    all_assignments,
                    np.array(all_scores),
                    phase_order,
                    feature_config,
                )
                if quality_or_none is not None:
                    quality_delta = (
                        quality_or_none - quality_history[-1]
                        if quality_history
                        else quality_or_none
                    )
                    # NOTE: quality_history may reflect different feature configs at
                    # different checkpoints if expansion was upgraded mid-loop.
                    # A jump is real progress and the plateau detector correctly
                    # sees it as such.
                    quality_history.append(quality_or_none)
                    samples_at_check.append(len(all_assignments))
                    logger.info(
                        "Quality=%.4g (%+.4g) at %d samples "
                        "[order=%d, range=%d]",
                        quality_or_none,
                        quality_delta,
                        len(all_assignments),
                        feature_config.max_order,
                        feature_config.pairwise_max_range,
                    )
                else:
                    # Bayesian stash-only after expansion change
                    logger.info(
                        "Bayesian: re-initialized posterior checkpoint (%d samples)",
                        len(all_assignments),
                    )
        finally:
            progress_file.close()
            snowball_file.close()
            try:
                write_meta(
                    progress_path,
                    progress_meta.model_copy(
                        update={"num_records": len(all_assignments) - n_supplemental}
                    ),
                )
                write_meta(
                    snowball_path,
                    snowball_meta.model_copy(
                        update={"num_records": len(all_assignments)}
                    ),
                )
            except Exception:
                logger.warning("Failed to update sidecar metadata", exc_info=True)

        # Build result dataset
        if not quality_history:
            raise CommandError(
                "No quality scores computed — increase max_samples beyond "
                f"min_samples ({self.config.min_samples}) to allow at least "
                "two quality checkpoints for Bayesian surrogate."
            )
        dataset = ExplorationDataset(
            assignments=all_assignments,
            scores=all_scores,
            objective_name=obj_name,
            num_layers=num_layers,
            mixer_names=phase_order,
            num_samples=len(all_assignments),
            termination_reason=termination_reason,
            feature_config=feature_config,
            surrogate_config=self.config.surrogate,
            expansion_result=last_expansion_result,
            quality_history=quality_history,
            samples_at_check=samples_at_check,
            final_quality=quality_history[-1] if quality_history else 0.0,
        )

        # Save using result handler
        result_handler = self.config.result_handler.create_result_handler()
        await result_handler.save(
            dataset,
            objective_result_path(output_dir, "exploration_results", obj_name),
        )

        logger.info(
            "Exploration complete for %s: %d/%d samples, "
            "final quality=%.4g, stopped=%s, surrogate=%s",
            obj_name,
            dataset.num_samples,
            self.config.max_samples,
            dataset.final_quality,
            dataset.termination_reason,
            self.config.surrogate.surrogate_type,
        )

    def _check_quality(
        self,
        assignments: list[tuple[str, ...]],
        scores: np.ndarray,
        mixer_names: list[str],
        feature_config: FeatureExpansionConfig,
    ) -> float | None:
        """Compute surrogate quality metric, dispatching by surrogate type.

        Returns float for a real quality value, or None for "stash only"
        (Bayesian first checkpoint — no previous posterior to score against).
        """
        match self.config.surrogate:
            case BayesianRidgeSurrogateConfig():
                return self._bayesian_quality_check(
                    assignments, scores, mixer_names, feature_config,
                )
            case RidgeSurrogateConfig() as cfg:
                return ridge_cv_rho(
                    assignments=assignments,
                    scores=scores,
                    mixer_names=mixer_names,
                    feature_config=feature_config,
                    cv_folds=cfg.cv_folds,
                    seed=self.config.seed,
                )
            case other:
                assert_never(other)

    def _bayesian_quality_check(
        self,
        assignments: list[tuple[str, ...]],
        scores: np.ndarray,
        mixer_names: list[str],
        feature_config: FeatureExpansionConfig,
    ) -> float | None:
        """Bayesian batch posterior predictive quality check.

        First call (no stashed surrogate): fit and stash, return None.
        Subsequent calls: score new batch under old posterior, refit in-place.
        """
        from place_layers.surrogate.bayesian_ridge import BayesianRidgeSurrogate

        cfg = self.config.surrogate
        assert isinstance(cfg, BayesianRidgeSurrogateConfig)

        if self._prev_surrogate is None:
            surrogate = cfg.create_surrogate(
                mixer_names=mixer_names,
                seed=self.config.seed,
                feature_config=feature_config,
            )
            if cfg.alpha_selection == "evidence":
                selected_alpha = surrogate.fit_optimize_alpha(assignments, scores)
                logger.info("Type-II ML alpha: %.4g", selected_alpha)
            else:
                surrogate.fit(assignments, scores)
            self._prev_surrogate = surrogate
            return None

        assert isinstance(self._prev_surrogate, BayesianRidgeSurrogate)
        prev_n = self._prev_surrogate.n_samples
        quality = float(np.mean(
            self._prev_surrogate.predictive_logpdf(
                assignments[prev_n:], scores[prev_n:]
            )
        ))

        # Refit in-place on all data for next checkpoint.
        # fit() is idempotent — fully replaces posterior state.
        self._prev_surrogate.fit(assignments, scores)
        return quality

    def _check_target_reached(self, quality: float) -> bool:
        """Check if quality target is reached (Ridge only; Bayesian has no target)."""
        match self.config.surrogate:
            case BayesianRidgeSurrogateConfig():
                return False
            case RidgeSurrogateConfig() as cfg:
                return quality >= cfg.target_rank_correlation
            case other:
                assert_never(other)

    def _min_samples_for_quality_check(self) -> int:
        """Minimum samples before the first quality check."""
        match self.config.surrogate:
            case BayesianRidgeSurrogateConfig():
                return self.config.min_samples  # One full batch
            case RidgeSurrogateConfig() as cfg:
                return cfg.cv_folds
            case other:
                assert_never(other)

    def _select_expansion(
        self,
        assignments: list[tuple[str, ...]],
        scores: np.ndarray,
        mixer_names: list[str],
    ) -> ExpansionSelectionResult:
        """Dispatch expansion selection based on surrogate type.

        Bayesian Ridge surrogate → evidence-based selection.
        Ridge surrogate → RidgeCV Spearman.
        """
        match self.config.surrogate:
            case BayesianRidgeSurrogateConfig() as cfg:
                return select_expansion_bayes_evidence(
                    assignments=assignments,
                    scores=scores,
                    mixer_names=mixer_names,
                    prior_alpha=cfg.prior_alpha,
                    prior_a0=cfg.prior_a0,
                    prior_b0=cfg.prior_b0,
                    feature_scaling=cfg.feature_scaling,
                    max_pairwise_range=self.config.expansion_max_range,
                    max_order=self.config.expansion_max_order,
                )
            case RidgeSurrogateConfig() as cfg:
                return select_expansion_ridge_cv(
                    assignments=assignments,
                    scores=scores,
                    mixer_names=mixer_names,
                    cv_folds=cfg.cv_folds,
                    target_rho=cfg.target_rank_correlation,
                    max_pairwise_range=self.config.expansion_max_range,
                    max_order=self.config.expansion_max_order,
                    seed=self.config.seed,
                )
            case other:
                assert_never(other)


# ============================================================================
# Config
# ============================================================================


class ExplorePlacementsCommandConfig(BaseCommandConfig):
    """Configuration for explore_placements command.

    All defaults are in the YAML config file (conf/command/explore_placements.yaml).
    """

    _impl_class: ClassVar[type[BaseCommand]] = ExplorePlacementsCommand

    command: Literal["explore_placements"] = "explore_placements"

    @model_validator(mode="before")
    @classmethod
    def _reject_deprecated_keys(cls, data: Any) -> Any:
        deprecated = {"cv_folds", "target_rank_correlation"}
        found = deprecated & set(data) if isinstance(data, dict) else set()
        if found:
            raise ValueError(
                f"Deprecated top-level key(s) {found}. "
                f"Move into surrogate config (Ridge only)."
            )
        return data

    backend: BackendConfig | None = Field(
        default=None,
        description="Backend configuration. None for mock evaluators.",
    )

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
    plateau_patience: int = Field(
        description="Stop if quality doesn't improve for this many batches.",
        gt=0,
    )
    plateau_min_improvement: float = Field(
        description="Minimum quality improvement to count as progress.",
        ge=0,
    )

    seed: int = Field(description="Random seed for reproducibility.")

    surrogate: SurrogateConfig = Field(
        description="Surrogate model configuration (ridge or bayesian_ridge). "
        "Determines expansion selection method: Bayesian evidence for bayesian_ridge, "
        "RidgeCV Spearman for ridge.",
    )

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
    fixed_expansion_config: FeatureExpansionConfig | None = Field(
        default=None,
        description=(
            "If set, use this feature expansion config directly instead of "
            "running automatic expansion selection. Useful when you already "
            "know the desired order and range (e.g., order=3, range=1 for "
            "adjacent triplets)."
        ),
    )
    supplemental_data_dirs: list[str] = Field(
        description=(
            "Directories containing pre-existing evaluation data to merge "
            "before exploration begins. Each directory is searched recursively "
            "for evaluation record files (evaluation_records.jsonl or legacy "
            "evaluations_{obj}.jsonl). Supplemental records are deduplicated "
            "against locally-generated progress but do not advance the RNG, "
            "preserving deterministic resumption."
        ),
    )
