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

import logging
import random
from pathlib import Path
from typing import ClassVar, Literal, assert_never

import numpy as np
from hydra.core.hydra_config import HydraConfig
from pydantic import Field

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
    EvaluationLogMeta,
    EvaluationRecord,
    append_record,
    load_records,
    write_meta,
)
from place_layers.surrogate.encoding import FeatureExpansionConfig, compute_feature_dim
from place_layers.surrogate.config import BayesianRidgeSurrogateConfig, RidgeSurrogateConfig, SurrogateConfig
from place_layers.surrogate.model_selection import (
    ExpansionSelectionResult,
    ridge_cv_rho,
    select_expansion_bayes_evidence,
    select_expansion_ridge_cv,
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
    if not path.exists():
        return [], [], []

    records = load_records(path)
    if records:
        logger.info("Loaded %d samples from %s", len(records), path)
    return (
        [r.assignment for r in records],
        [r.score for r in records],
        [r.allocation for r in records],
    )


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
        lines.append(
            f"  Phase 2 (order at range={result.selected_range}):"
        )
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

    async def run(self) -> None:
        """Execute adaptive exploration loop."""
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)

        logger.info("Output directory: %s", output_dir)
        logger.info("Starting placement exploration...")

        from contextlib import nullcontext

        backend_cm = self.config.backend.create_backend() if self.config.backend else nullcontext()
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

        # Load existing progress (per-objective JSONL)
        progress_path = output_dir / f"exploration_progress_{obj_name}.jsonl"
        all_assignments, all_scores, all_allocations = _load_progress(
            progress_path
        )

        # Capture local count before merging supplemental data
        num_local = len(all_assignments)
        n_supplemental = 0

        if self.config.supplemental_data_dirs:
            from place_layers.surrogate.evaluation_log import (
                load_pooled_data,
                resolve_supplemental_dirs,
            )
            supplemental_files = resolve_supplemental_dirs(
                self.config.supplemental_data_dirs, obj_name
            )
            if supplemental_files:
                pool = load_pooled_data(supplemental_files, deduplicate=True)
                existing = {(a, s) for a, s in zip(all_assignments, all_scores)}
                n_before = len(all_assignments)
                for a, s, alloc in zip(pool.assignments, pool.scores, pool.allocations):
                    if (a, s) not in existing:
                        all_assignments.append(a)
                        all_scores.append(s)
                        all_allocations.append(alloc)
                        existing.add((a, s))
                n_added = len(all_assignments) - n_before
                n_supplemental = n_added
                logger.info(
                    "Supplemental data: loaded %d records from %d file(s), "
                    "merged %d new (skipped %d duplicates) for objective '%s'",
                    pool.num_samples, len(supplemental_files),
                    n_added, pool.num_samples - n_added, obj_name,
                )
            else:
                logger.info(
                    "Supplemental data: no files found for objective '%s' in %d director%s",
                    obj_name, len(self.config.supplemental_data_dirs),
                    "y" if len(self.config.supplemental_data_dirs) == 1 else "ies",
                )

        rho_history: list[float] = []
        samples_at_check: list[int] = []

        # Create RNG and fast-forward past completed samples
        rng = random.Random(self.config.seed)
        if num_local > 0:
            _fast_forward_rng(
                rng, num_local, feasible_allocations, num_layers,
                weights=sampling_weights,
            )
            if len(all_assignments) > num_local:
                logger.info(
                    "Resumed from %d local samples (%d total with supplemental)",
                    num_local, len(all_assignments),
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

        # Open progress file for incremental writes
        progress_file = progress_path.open("a")
        try:
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
                    record = EvaluationRecord(
                        assignment=assignment_tuple, score=score, allocation=allocation,
                    )
                    append_record(progress_file, record)

            # --- Phase 2: Initial expansion selection ---
            # Expansion selection is re-run in the adaptive loop (Phase 3) after
            # each batch. The feature-to-sample guard in _select_expansion()
            # relaxes as more data arrives, potentially unlocking higher-order
            # features. Re-running is cheap (a few surrogate fits).
            feature_config = FeatureExpansionConfig.unary()
            last_expansion_result: ExpansionSelectionResult | None = None
            if (
                self.config.expansion_max_order > 1
                and len(all_assignments) >= self.config.cv_folds
            ):
                expansion_result = self._select_expansion(
                    all_assignments, np.array(all_scores), phase_order,
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
                    record = EvaluationRecord(
                        assignment=assignment_tuple, score=score, allocation=allocation,
                    )
                    append_record(progress_file, record)

                # Re-run expansion selection with larger dataset.
                # The feature-to-sample guard relaxes as data grows,
                # potentially unlocking higher-order features.
                if (
                    self.config.expansion_max_order > 1
                    and len(all_assignments) >= self.config.cv_folds
                ):
                    expansion_result = self._select_expansion(
                        all_assignments, np.array(all_scores), phase_order,
                    )
                    last_expansion_result = expansion_result
                    new_config = expansion_result.selected_config
                    if new_config != feature_config:
                        old_feature_config = feature_config
                        feature_config = new_config
                        _log_expansion_result(
                            expansion_result,
                            n_samples=len(all_assignments),
                            num_types=len(phase_order),
                            num_layers=num_layers,
                            surrogate_config=self.config.surrogate,
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
                logger.info(
                    "Ridge CV rho=%.4g (%+.4g) at %d samples "
                    "[order=%d, range=%d]",
                    rho,
                    rho_delta,
                    len(all_assignments),
                    feature_config.max_order,
                    feature_config.pairwise_max_range,
                )
        finally:
            progress_file.close()

        # Update sidecar with final record count
        write_meta(
            progress_path,
            progress_meta.model_copy(update={"num_records": len(all_assignments) - n_supplemental}),
        )

        # Build result dataset
        assert rho_history, (
            "Quality check was never run — need at least cv_folds samples"
        )
        dataset = ExplorationDataset(
            assignments=all_assignments,
            scores=all_scores,
            allocations=all_allocations,
            objective_name=obj_name,
            num_layers=num_layers,
            mixer_names=phase_order,
            num_samples=len(all_assignments),
            termination_reason=termination_reason,
            feature_config=feature_config,
            surrogate_config=self.config.surrogate,
            expansion_result=last_expansion_result,
            rho_history=rho_history,
            samples_at_check=samples_at_check,
            final_rho=rho_history[-1] if rho_history else 0.0,
        )

        # Save using result handler
        result_handler = self.config.result_handler.create_result_handler()
        await result_handler.save(
            dataset,
            objective_result_path(output_dir, "exploration_results", obj_name),
        )

        logger.info(
            "Exploration complete for %s: %d/%d samples, "
            "final rho=%.4g, stopped=%s, surrogate=%s",
            obj_name,
            dataset.num_samples,
            self.config.max_samples,
            dataset.final_rho,
            dataset.termination_reason,
            self.config.surrogate.surrogate_type,
        )

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
            case RidgeSurrogateConfig():
                return select_expansion_ridge_cv(
                    assignments=assignments,
                    scores=scores,
                    mixer_names=mixer_names,
                    cv_folds=self.config.cv_folds,
                    target_rho=self.config.target_rank_correlation,
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
            "before exploration begins. Each directory is searched per-objective "
            "for exploration_progress_{obj}.jsonl and evaluations_{obj}.jsonl files. "
            "Supplemental records are deduplicated against locally-generated progress "
            "but do not advance the RNG, preserving deterministic resumption."
        ),
    )
