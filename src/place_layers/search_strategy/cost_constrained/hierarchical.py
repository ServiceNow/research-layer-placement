"""Hierarchical beam search for cost-constrained optimization.

This implements an inner loop strategy that uses the hierarchical beam search
approach: phase-by-phase assignment of layers to mixers.
"""

from __future__ import annotations

import logging
import math
from math import comb
from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import Field

from place_layers.core.types import LayerPlacement
from place_layers.cost_model.base_cost_model import Allocation, BaseCostModel
from place_layers.problem.layer_placement import CostConstrainedProblem
from place_layers.search_strategy.beam_core import (
    create_layer_assignment,
    generate_beam_candidates,
    select_top_beam,
)
from place_layers.search_strategy.cost_constrained.base import BaseCostConstrainedSearch
from place_layers.search_strategy.cost_constrained.base_config import (
    BaseCostConstrainedConfig,
)
from place_layers.search_strategy.cost_constrained.types import BeamEntry

if TYPE_CHECKING:
    from place_layers.problem.base_problem import BaseProblem
    from place_layers.search_strategy.base_search_strategy import BaseSearchStrategy
    from place_layers.search_strategy.progress import SearchObserver
    from place_layers.search_strategy.progress.estimator import SearchProgressEstimator


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Statistical bounds calculation for progress estimation
# -----------------------------------------------------------------------------


def continuation_rate(early_stop_threshold: float) -> float:
    """Probability of continuing to the next growth step.

    Models the early stopping behavior probabilistically. Each growth step
    has a probability of being skipped due to early stopping (no score improvement).

    Empirically calibrated from observed search behavior. The rate varies with
    threshold but also with problem-specific factors (noise, score landscape).

    Args:
        early_stop_threshold: The early_stop_threshold config parameter.

    Returns:
        Probability in [0, 1] of continuing to the next growth step.
    """
    # Two-regime model:
    # - threshold=0.0: rate = 0.98 (nearly all steps run, only natural stopping)
    # - threshold>0: rate decays more rapidly with threshold
    #
    # For threshold=0.0: early stopping only triggers when score doesn't improve
    # at all, which is rare with noisy evaluations.
    #
    # For threshold=0.001: early stopping triggers more often, reducing steps run.
    #
    # Model: rate = 0.98 * exp(-50 * threshold)
    # - threshold=0.0: rate = 0.98
    # - threshold=0.001: rate = 0.98 * 0.951 = 0.932
    # - threshold=0.01: rate = 0.98 * 0.607 = 0.595
    base_rate = 0.98
    k = 50.0
    return base_rate * math.exp(-k * early_stop_threshold)


def phase_eval_statistics(
    budget: int,
    unassigned: int,
    beam_width: int,
    cont_rate: float,
) -> tuple[float, float]:
    """Compute expected evaluations and variance for a phase.

    Uses probabilistic model where P(run step s) = adjusted_rate^s.
    The continuation rate is adjusted upward for large budgets because
    phases with many layers to assign have more room for score improvement,
    making early stopping less likely.

    Args:
        budget: Number of layers to assign in this phase.
        unassigned: Number of unassigned layers at phase start.
        beam_width: Beam width parameter (B).
        cont_rate: Base continuation rate from continuation_rate().

    Returns:
        Tuple of (expected_evals, variance) for this phase.
    """
    if budget <= 0:
        return 0.0, 0.0
    if budget == 1:
        # Prescore only, exact count
        return float(unassigned), 0.0

    # Adjust continuation rate for large budgets.
    # Large budgets have more room for improvement, so early stopping is less likely.
    # Blend base rate towards 0.99 as budget increases (saturates around budget=20).
    # - budget=5:  factor=0.25, adjusted ≈ cont_rate + 0.25*(0.99-cont_rate)
    # - budget=20: factor=1.0,  adjusted ≈ 0.99
    # - budget=48: factor=1.0,  adjusted ≈ 0.99
    budget_factor = min(budget / 20.0, 1.0)
    high_rate = 0.99
    adjusted_rate = cont_rate + budget_factor * (high_rate - cont_rate)

    # Prescore is exact (no variance)
    expected = float(unassigned)
    variance = 0.0

    # Growth steps are probabilistic
    for s in range(1, budget):
        p_run = adjusted_rate**s  # P(reach and run step s)
        evals_if_run = min(beam_width * (unassigned - s), comb(unassigned, s + 1))

        # Expected contribution: p * evals
        expected += p_run * evals_if_run

        # Variance contribution: p * (1-p) * evals^2 (Bernoulli variance)
        variance += p_run * (1 - p_run) * (evals_if_run**2)

    return expected, variance


def allocation_statistics(
    budgets: list[int],
    num_layers: int,
    beam_width: int,
    cross_phase_beam_width: int,
    cont_rate: float,
) -> tuple[float, float]:
    """Calculate expected evaluations and variance for a single allocation.

    Aggregates phase statistics across all phases. Since phases are
    independent, variances add.

    Note: The cross_phase_beam_width multiplier applies to phases AFTER the
    first non-zero phase, not after the first phase in the list. Phases with
    budget=0 are skipped entirely and don't count.

    Model uncertainty: The continuation rate model is approximate. To ensure
    bounds contain actual values, we add model uncertainty variance proportional
    to expected^2, giving ~15% uncertainty in addition to the step-by-step variance.

    Args:
        budgets: Phase budgets (layers to assign in each phase, excluding last).
        num_layers: Total number of layers (N).
        beam_width: Within-phase beam width (B).
        cross_phase_beam_width: Cross-phase beam width (C).
        cont_rate: Continuation rate from continuation_rate().

    Returns:
        Tuple of (expected_evals, variance) for this allocation.
    """
    # Final evaluation always happens (exact, no variance)
    expected = 1.0
    variance = 0.0
    unassigned = num_layers
    first_nonzero_seen = False

    for budget in budgets:
        if budget == 0:
            continue

        # First non-zero phase has multiplier 1, subsequent phases use cross_phase_beam_width
        if first_nonzero_seen:
            multiplier = cross_phase_beam_width
        else:
            multiplier = 1
            first_nonzero_seen = True

        phase_exp, phase_var = phase_eval_statistics(
            budget, unassigned, beam_width, cont_rate
        )

        # Scale by multiplier (for cross-phase beam width)
        expected += multiplier * phase_exp
        # Variance scales by multiplier^2 when scaling a random variable
        variance += (multiplier**2) * phase_var

        unassigned -= budget

    # Add model uncertainty: ~30% coefficient of variation
    # This accounts for problem-specific factors not captured by the model
    # (score landscape, noise characteristics, interaction effects)
    model_uncertainty_cv = 0.30
    model_uncertainty_var = (model_uncertainty_cv * expected) ** 2
    variance += model_uncertainty_var

    return expected, variance


class HierarchicalSearch(BaseCostConstrainedSearch):
    """
    Cost-constrained search using hierarchical beam search for the inner loop.

    Within each allocation, uses the hierarchical approach:
    - Phase 1: Find best layers for mixer 0 (e.g., FA)
    - Phase 2: Given fixed mixer 0 positions, find best layers for mixer 1 (e.g., SWA)
    - Phase 3: Remaining layers get mixer 2 (e.g., GDN)

    This is much faster than swap-based search (~10x fewer evaluations) but
    has a different search bias - it cannot reconsider early phase decisions.

    Key difference from standalone HierarchicalBeamSearch:
    - This is an inner loop within cost-constrained search
    - Allocation is fixed (counts per mixer already determined)
    - Only assignment (which layers) needs to be found
    """

    def __init__(
        self,
        config: "HierarchicalConfig",
        observer: "SearchObserver | None" = None,
        progress_estimator: "SearchProgressEstimator | None" = None,
    ) -> None:
        super().__init__(config, observer=observer, progress_estimator=progress_estimator)
        self.config: HierarchicalConfig

    async def _search_within_allocation(
        self,
        problem: CostConstrainedProblem,
        cost_model: BaseCostModel,
        allocation: Allocation,
        num_layers: int,
    ) -> BeamEntry | None:
        """
        Search for the best assignment within a fixed allocation using hierarchical beam search.

        Runs phases sequentially, each phase finding the best layers for one mixer type.
        Phase order is determined by the order of keys in mixer_costs config, not alphabetical.

        When cross_phase_beam_width > 1, maintains multiple candidate paths across phases,
        allowing exploration of phase interactions at higher computational cost.
        """
        alloc_cost = cost_model.allocation_cost(allocation)
        mixer_names = problem.evaluator.mixer_names  # Canonical order for allocations
        phase_order = cost_model.phase_order  # User-specified order for phases
        cross_phase_k = self.config.cross_phase_beam_width

        # Build mapping from mixer name to allocation index
        name_to_alloc_idx = {name: i for i, name in enumerate(mixer_names)}

        # Start with a single path
        # fixed_assignments maps layer_idx -> mixer_name
        paths: list[dict] = [
            {
                "fixed": {},
                "unassigned": list(range(num_layers)),
                "score": 0.0,
            }
        ]

        # Build phase budgets in phase_order (excluding the last which gets remainder)
        # Each phase assigns layers to one mixer type
        phase_budgets = [
            (phase_order[i], allocation[name_to_alloc_idx[phase_order[i]]])
            for i in range(len(phase_order) - 1)
        ]

        # Total phases is the number of phases we'll actually run (non-zero budgets)
        total_phases = sum(1 for _, b in phase_budgets if b > 0)
        running_phase_idx = 0

        # Run phases for each mixer type except the last (in phase_order)
        for phase_idx, (mixer_name, budget) in enumerate(phase_budgets):
            fallback_mixer = phase_order[phase_idx + 1]

            if budget == 0:
                # Skip phases with no layers to assign
                continue

            running_phase_idx += 1

            # Emit phase started event
            if self.observer is not None:
                from place_layers.search_strategy.progress import PhaseStartedEvent

                self.observer.on_phase_started(
                    PhaseStartedEvent(
                        phase_index=running_phase_idx,
                        total_phases=total_phases,
                        mixer_name=mixer_name,
                        budget=budget,
                        num_paths=len(paths),
                    )
                )

            new_paths: list[dict] = []

            for path_idx, path in enumerate(paths):
                if not path["unassigned"]:
                    # No more layers to assign for this path
                    continue

                # Emit path started event for cross-phase visibility
                if self.observer is not None:
                    from place_layers.search_strategy.progress import PathStartedEvent

                    # Compute fixed_by_mixer from path["fixed"]
                    fixed_by_mixer: dict[str, list[int]] = {}
                    for layer_idx, mixer in path["fixed"].items():
                        if mixer not in fixed_by_mixer:
                            fixed_by_mixer[mixer] = []
                        fixed_by_mixer[mixer].append(layer_idx)
                    # Convert to sorted tuples
                    fixed_by_mixer_tuples = {
                        k: tuple(sorted(v)) for k, v in fixed_by_mixer.items()
                    }

                    self.observer.on_path_started(
                        PathStartedEvent(
                            phase_index=running_phase_idx,
                            path_index=path_idx,
                            total_paths=len(paths),
                            inherited_score=path["score"],
                            fixed_by_mixer=fixed_by_mixer_tuples,
                        )
                    )

                # Find best layers for this mixer, given this path's state
                beam = await self._run_phase(
                    problem=problem,
                    budget=budget,
                    mixer_name=mixer_name,
                    fallback_mixer=fallback_mixer,
                    unassigned=path["unassigned"],
                    fixed_assignments=path["fixed"],
                    num_layers=num_layers,
                    phase_index=running_phase_idx,
                )

                # Create new paths for each beam entry
                for entry in beam:
                    new_fixed = dict(path["fixed"])
                    new_unassigned = list(path["unassigned"])
                    for layer_idx in entry["layers"]:
                        new_fixed[layer_idx] = mixer_name
                        new_unassigned.remove(layer_idx)
                    new_paths.append(
                        {
                            "fixed": new_fixed,
                            "unassigned": new_unassigned,
                            "score": entry["score"],
                        }
                    )

            # Prune to top cross_phase_k paths
            new_paths.sort(key=lambda p: p["score"], reverse=True)
            paths = new_paths[:cross_phase_k]

            if not paths:
                # No valid paths remain
                return None

            # Emit phase completed event
            if self.observer is not None:
                from place_layers.search_strategy.progress import PhaseCompletedEvent

                # Get the layers assigned in this phase from the best path
                best_path_after = paths[0]
                phase_layers = sorted(
                    [
                        layer_idx
                        for layer_idx, mixer in best_path_after["fixed"].items()
                        if mixer == mixer_name
                    ]
                )

                self.observer.on_phase_completed(
                    PhaseCompletedEvent(
                        phase_index=running_phase_idx,
                        total_phases=total_phases,
                        mixer_name=mixer_name,
                        budget=budget,
                        final_layers=tuple(phase_layers),
                        final_score=best_path_after["score"],
                    )
                )

        # Take best path for final assignment
        best_path = paths[0]

        # Assign remaining layers to the last mixer (last in phase_order)
        last_mixer = phase_order[-1]
        for layer_idx in best_path["unassigned"]:
            best_path["fixed"][layer_idx] = last_mixer

        # Build final assignment list
        assignments = [best_path["fixed"][i] for i in range(num_layers)]

        # Final evaluation
        placement = LayerPlacement(assignments=tuple(assignments))
        final_score = await problem.evaluate(placement)
        self.eval_count += 1

        return BeamEntry(
            assignments=tuple(assignments),
            allocation=allocation,
            cost=alloc_cost,
            score=final_score,
        )

    async def _run_phase(
        self,
        problem: CostConstrainedProblem,
        budget: int,
        mixer_name: str,
        fallback_mixer: str,
        unassigned: list[int],
        fixed_assignments: dict[int, str],
        num_layers: int,
        phase_index: int,
    ) -> list[dict]:
        """
        Run one phase: find best 'budget' layers for mixer_name.

        Returns beam of candidates, each with "layers" and "score" keys.
        """
        # Pre-score individual layers
        layer_scores = await self._prescore_layers(
            problem=problem,
            unassigned=unassigned,
            mixer_name=mixer_name,
            fallback_mixer=fallback_mixer,
            fixed_assignments=fixed_assignments,
            num_layers=num_layers,
        )

        if budget == 1:
            # Simple case: just take the best single layer
            best_layer, best_score = layer_scores[0]
            beam = [{"layers": [best_layer], "score": best_score, "parent_idx": 0, "added_layer": best_layer}]

            # Emit beam growth step for single-layer phases too
            if self.observer is not None:
                from place_layers.search_strategy.progress import BeamGrowthStepEvent

                self.observer.on_beam_growth_step(
                    BeamGrowthStepEvent(
                        step=1,
                        total_steps=1,
                        phase_index=phase_index,
                        mixer_name=mixer_name,
                        parents=({"layers": [], "score": None},),
                        tried=tuple(
                            {"layer": layer, "score": score, "parent_idx": 0}
                            for layer, score in layer_scores
                        ),
                        beam=tuple(
                            {"layers": b["layers"], "score": b["score"], "parent_idx": 0, "added_layer": b["layers"][0]}
                            for b in beam[:5]
                        ),
                    )
                )

            return beam

        # Grow beam from best single layers to budget size
        beam = await self._grow_beam(
            problem=problem,
            layer_scores=layer_scores,
            budget=budget,
            mixer_name=mixer_name,
            fallback_mixer=fallback_mixer,
            unassigned=unassigned,
            fixed_assignments=fixed_assignments,
            num_layers=num_layers,
            phase_index=phase_index,
        )

        return beam

    async def _prescore_layers(
        self,
        problem: CostConstrainedProblem,
        unassigned: list[int],
        mixer_name: str,
        fallback_mixer: str,
        fixed_assignments: dict[int, str],
        num_layers: int,
    ) -> list[tuple[int, float]]:
        """Pre-score individual layers to seed the beam."""
        layer_scores = []

        for layer_idx in unassigned:
            assignments = create_layer_assignment(
                fixed_assignments=fixed_assignments,
                target_layers=[layer_idx],
                target_mixer=mixer_name,
                unassigned=unassigned,
                fallback_mixer=fallback_mixer,
                num_layers=num_layers,
            )
            placement = LayerPlacement(assignments=tuple(assignments))
            score = await problem.evaluate(placement)
            self.eval_count += 1

            # Notify observer of eval completion
            if self.observer is not None:
                self.observer.on_eval_completed(self.eval_count)

            layer_scores.append((layer_idx, score))

        # Sort by score (higher is better)
        layer_scores.sort(key=lambda x: x[1], reverse=True)
        return layer_scores

    async def _grow_beam(
        self,
        problem: CostConstrainedProblem,
        layer_scores: list[tuple[int, float]],
        budget: int,
        mixer_name: str,
        fallback_mixer: str,
        unassigned: list[int],
        fixed_assignments: dict[int, str],
        num_layers: int,
        phase_index: int,
    ) -> list[dict]:
        """Grow the beam from seed to budget size."""
        # Initialize beam with single-layer candidates
        beam = [
            {"layers": [layer_idx], "score": score}
            for layer_idx, score in layer_scores[: self.config.beam_width]
        ]

        best_score = beam[0]["score"]

        total_growth_steps = budget - 1

        # Emit step 1 event for initial beam (from prescoring)
        if self.observer is not None:
            from place_layers.search_strategy.progress import BeamGrowthStepEvent

            self.observer.on_beam_growth_step(
                BeamGrowthStepEvent(
                    step=1,
                    total_steps=total_growth_steps,
                    phase_index=phase_index,
                    mixer_name=mixer_name,
                    parents=({"layers": [], "score": None},),
                    tried=tuple(
                        {"layer": layer, "score": score, "parent_idx": 0}
                        for layer, score in layer_scores
                    ),
                    beam=tuple(
                        {"layers": b["layers"], "score": b["score"], "parent_idx": 0, "added_layer": b["layers"][0]}
                        for b in beam[:5]
                    ),
                )
            )

        # Grow beam iteratively
        for growth_step in range(1, budget):
            # Generate candidates using shared function
            candidates = generate_beam_candidates(beam, unassigned)

            if not candidates:
                break

            # Evaluate candidates
            await self._evaluate_candidates(
                problem=problem,
                candidates=candidates,
                mixer_name=mixer_name,
                fallback_mixer=fallback_mixer,
                unassigned=unassigned,
                fixed_assignments=fixed_assignments,
                num_layers=num_layers,
            )

            # Select top candidates using shared function (higher is better)
            new_beam = select_top_beam(
                candidates, self.config.beam_width, higher_is_better=True
            )

            # Emit beam growth step event with expansion details
            if self.observer is not None:
                from place_layers.search_strategy.progress import BeamGrowthStepEvent

                # Build tried list with provenance from candidates
                tried = [
                    {"layer": c["added_layer"], "score": c["score"], "parent_idx": c["parent_idx"]}
                    for c in candidates
                ]
                # Sort by score descending for display
                tried.sort(key=lambda x: x["score"], reverse=True)

                # Build parents list from previous beam
                parents = tuple(
                    {"layers": b["layers"], "score": b["score"]}
                    for b in beam
                )

                # Build beam with provenance
                beam_with_provenance = tuple(
                    {"layers": b["layers"], "score": b["score"], "parent_idx": b["parent_idx"], "added_layer": b["added_layer"]}
                    for b in new_beam[:5]
                )

                self.observer.on_beam_growth_step(
                    BeamGrowthStepEvent(
                        step=growth_step + 1,  # +1 because step 1 was prescoring
                        total_steps=total_growth_steps,
                        phase_index=phase_index,
                        mixer_name=mixer_name,
                        parents=parents,
                        tried=tuple(tried),
                        beam=beam_with_provenance,
                    )
                )

            beam = new_beam

            # Check early stopping
            new_best_score = beam[0]["score"]
            if new_best_score <= best_score + self.config.early_stop_threshold:
                break
            best_score = new_best_score

        return beam

    async def _evaluate_candidates(
        self,
        problem: CostConstrainedProblem,
        candidates: list[dict],
        mixer_name: str,
        fallback_mixer: str,
        unassigned: list[int],
        fixed_assignments: dict[int, str],
        num_layers: int,
    ) -> None:
        """Evaluate all candidates and store scores."""
        for candidate in candidates:
            assignments = create_layer_assignment(
                fixed_assignments=fixed_assignments,
                target_layers=candidate["layers"],
                target_mixer=mixer_name,
                unassigned=unassigned,
                fallback_mixer=fallback_mixer,
                num_layers=num_layers,
            )
            placement = LayerPlacement(assignments=tuple(assignments))
            candidate["score"] = await problem.evaluate(placement)
            self.eval_count += 1

            # Notify observer of eval completion
            if self.observer is not None:
                self.observer.on_eval_completed(self.eval_count)


class HierarchicalConfig(BaseCostConstrainedConfig):
    """Configuration for hierarchical beam search inner loop."""

    _impl_class: ClassVar[type["BaseSearchStrategy"]] = HierarchicalSearch

    strategy_type: Literal["cost_constrained_hierarchical"] = (
        "cost_constrained_hierarchical"
    )

    beam_width: int = Field(
        description="Number of top candidates to keep at each growth step",
        gt=0,
    )

    early_stop_threshold: float = Field(
        description="Stop growing beam if best score improvement is below this threshold",
        ge=0.0,
    )

    cross_phase_beam_width: int = Field(
        description=(
            "Number of candidate paths to maintain across phases. "
            "Defaults to beam_width (via Hydra interpolation). "
            "Set to 1 for greedy (lock in best after each phase)."
        ),
        gt=0,
    )

    def create_progress_estimator(
        self, problem: "BaseProblem"
    ) -> "SearchProgressEstimator | None":
        """Create progress estimator for hierarchical beam search.

        Uses statistical bounds (expected ± 2*std_dev) instead of worst-case
        bounds. This gives much tighter predictions (~2% spread vs 8x spread).

        The probabilistic model accounts for early stopping by modeling each
        growth step as a Bernoulli random variable with P(run) = continuation_rate^step.

        Args:
            problem: The optimization problem (must be CostConstrainedProblem).

        Returns:
            Configured SearchProgressEstimator, or None if not applicable.
        """
        from place_layers.problem.layer_placement import (
            MaxPerformanceProblemConfig,
            MinCostProblemConfig,
        )
        from place_layers.search_strategy.progress.estimator import (
            AllocationProgressInfo,
            ProgressEstimate,
            SearchProgressEstimator,
        )

        # Only works with cost-constrained problems
        if not isinstance(problem, CostConstrainedProblem):
            return None

        config = problem.config
        cost_model = problem.cost_model
        num_layers = problem.evaluator.num_layers
        mixer_names = problem.evaluator.mixer_names
        phase_order = cost_model.phase_order

        # Build name-to-allocation-index mapping
        name_to_alloc_idx = {name: i for i, name in enumerate(mixer_names)}

        # Enumerate allocations in the same order as search will use
        allocations: list[tuple[float, tuple[int, ...]]] = []
        for alloc_cost, allocation in cost_model.allocations_by_cost(
            num_layers, direction="decreasing"
        ):
            # Filter by cost_budget for max_performance only
            match config:
                case MaxPerformanceProblemConfig(cost_budget=budget):
                    if alloc_cost > budget:
                        continue
                case MinCostProblemConfig():
                    pass  # No filtering - enumerate all allocations

            if len(allocations) >= self.max_allocations:
                break
            allocations.append((alloc_cost, allocation))

        if not allocations:
            return None

        # Compute continuation rate from early_stop_threshold
        cont_rate = continuation_rate(self.early_stop_threshold)

        # Compute statistics for each allocation
        total_expected = 0.0
        total_variance = 0.0
        allocation_info: list[AllocationProgressInfo] = []

        for alloc_cost, allocation in allocations:
            # Get phase budgets in phase_order (excluding last which gets remainder)
            budgets = tuple(
                allocation[name_to_alloc_idx[phase_order[i]]]
                for i in range(len(phase_order) - 1)
            )

            # Compute expected and variance using probabilistic model
            alloc_exp, alloc_var = allocation_statistics(
                budgets=list(budgets),
                num_layers=num_layers,
                beam_width=self.beam_width,
                cross_phase_beam_width=self.cross_phase_beam_width,
                cont_rate=cont_rate,
            )

            # Compute statistical bounds (95% CI)
            std_dev = math.sqrt(alloc_var) if alloc_var > 0 else 0.0
            lower = int(alloc_exp - 2 * std_dev)
            upper = int(alloc_exp + 2 * std_dev)

            total_expected += alloc_exp
            total_variance += alloc_var

            allocation_info.append(
                AllocationProgressInfo(
                    allocation=allocation,
                    budgets=budgets,
                    cost=alloc_cost,
                    expected=alloc_exp,
                    variance=alloc_var,
                    estimated_lower=lower,
                    estimated_upper=upper,
                )
            )

        # Compute total bounds from aggregated statistics
        total_std_dev = math.sqrt(total_variance) if total_variance > 0 else 0.0

        return SearchProgressEstimator(
            initial=ProgressEstimate(
                lower=int(total_expected - 2 * total_std_dev),
                upper=int(total_expected + 2 * total_std_dev),
                expected=int(total_expected),
                std_dev=total_std_dev,
            ),
            allocation_info=tuple(allocation_info),
        )
