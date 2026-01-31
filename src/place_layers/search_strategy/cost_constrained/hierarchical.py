"""Hierarchical beam search for cost-constrained optimization.

This implements an inner loop strategy that uses the hierarchical beam search
approach: phase-by-phase assignment of layers to mixers.
"""

from __future__ import annotations

import logging
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
    from place_layers.search_strategy.base_search_strategy import BaseSearchStrategy
    from place_layers.search_strategy.progress import SearchObserver


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Bounds calculation functions for progress estimation
# -----------------------------------------------------------------------------


def phase_eval_lower(budget: int, unassigned: int, beam_width: int) -> int:
    """Lower bound for phase evaluations: prescore + one growth step (if budget > 1).

    Lower bound assumes early stopping after the first growth step.

    Args:
        budget: Number of layers to assign in this phase.
        unassigned: Number of unassigned layers at phase start.
        beam_width: Beam width parameter (B).

    Returns:
        Minimum number of evaluations for this phase.
    """
    if budget <= 0:
        return 0
    if budget == 1:
        # Only prescore, no growth
        return max(0, unassigned)
    # Prescore (U) + one growth step (B * (U - 1))
    return unassigned + beam_width * (unassigned - 1)


def phase_eval_upper(budget: int, unassigned: int, beam_width: int) -> int:
    """Upper bound for phase evaluations: prescore + full growth without early stopping.

    Upper bound assumes no early stopping occurs, so all growth steps execute.

    Args:
        budget: Number of layers to assign in this phase.
        unassigned: Number of unassigned layers at phase start.
        beam_width: Beam width parameter (B).

    Returns:
        Maximum number of evaluations for this phase.
    """
    if budget <= 0:
        return 0
    if budget == 1:
        # Only prescore, no growth
        return max(0, unassigned)

    # Prescore + all growth steps
    # At growth step s (1-indexed), we have beam entries of size s
    # Each entry can add any of (U - s) remaining layers
    # But we cap at beam_width entries, so we evaluate at most B * (U - s)
    # candidates per step.
    # Also, we can't generate more candidates than C(U, s+1) (combinations)
    growth_evals = sum(
        min(beam_width * (unassigned - s), comb(unassigned, s + 1))
        for s in range(1, budget)
    )
    return unassigned + growth_evals


def allocation_bounds(
    budgets: list[int],
    num_layers: int,
    beam_width: int,
    cross_phase_beam_width: int,
) -> tuple[int, int]:
    """Calculate evaluation count bounds for a single allocation.

    Args:
        budgets: Phase budgets (layers to assign in each phase, excluding last).
        num_layers: Total number of layers (N).
        beam_width: Within-phase beam width (B).
        cross_phase_beam_width: Cross-phase beam width (C).

    Returns:
        Tuple of (lower_bound, upper_bound) for this allocation.
    """
    lower, upper = 1, 1  # Final evaluation always happens
    unassigned = num_layers

    for p, budget in enumerate(budgets):
        if budget == 0:
            continue
        # First phase has multiplier 1, subsequent phases use cross_phase_beam_width
        multiplier = 1 if p == 0 else cross_phase_beam_width
        lower += multiplier * phase_eval_lower(budget, unassigned, beam_width)
        upper += multiplier * phase_eval_upper(budget, unassigned, beam_width)
        unassigned -= budget

    return lower, upper


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
    ) -> None:
        super().__init__(config, observer=observer)
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
        num_mixers = len(allocation)
        cross_phase_k = self.config.cross_phase_beam_width

        # Build mapping from mixer name to allocation index
        name_to_alloc_idx = {name: i for i, name in enumerate(mixer_names)}

        # Start with a single path
        # fixed_assignments maps layer_idx -> mixer_name
        paths: list[dict] = [{
            "fixed": {},
            "unassigned": list(range(num_layers)),
            "score": 0.0,
        }]

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
                    )
                )

            new_paths: list[dict] = []

            for path in paths:
                if not path["unassigned"]:
                    # No more layers to assign for this path
                    continue

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
                    total_phases=total_phases,
                )

                # Create new paths for each beam entry
                for entry in beam:
                    new_fixed = dict(path["fixed"])
                    new_unassigned = list(path["unassigned"])
                    for layer_idx in entry["layers"]:
                        new_fixed[layer_idx] = mixer_name
                        new_unassigned.remove(layer_idx)
                    new_paths.append({
                        "fixed": new_fixed,
                        "unassigned": new_unassigned,
                        "score": entry["score"],
                    })

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
                phase_layers = sorted([
                    layer_idx for layer_idx, mixer in best_path_after["fixed"].items()
                    if mixer == mixer_name
                ])

                self.observer.on_phase_completed(
                    PhaseCompletedEvent(
                        phase_index=running_phase_idx,
                        total_phases=total_phases,
                        mixer_name=mixer_name,
                        budget=budget,
                        final_layers=phase_layers,
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
        placement = LayerPlacement(assignments=assignments)
        final_score = await problem.evaluate(placement)
        self.eval_count += 1

        return BeamEntry(
            assignments=assignments,
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
        total_phases: int,
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
            beam = [{"layers": [best_layer], "score": best_score}]

            # Emit beam growth step for single-layer phases too
            if self.observer is not None:
                from place_layers.search_strategy.progress import BeamGrowthStepEvent

                self.observer.on_beam_growth_step(
                    BeamGrowthStepEvent(
                        step=1,
                        total_steps=1,
                        phase_index=phase_index,
                        mixer_name=mixer_name,
                        base_layers=[],
                        tried=[{"layer": layer, "score": score} for layer, score in layer_scores],
                        beam=beam[:5],
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
            placement = LayerPlacement(assignments=assignments)
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
                    base_layers=[],
                    tried=[{"layer": layer, "score": score} for layer, score in layer_scores],
                    beam=beam[:5],
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
            new_beam = select_top_beam(candidates, self.config.beam_width, higher_is_better=True)

            # Emit beam growth step event with expansion details
            if self.observer is not None:
                from place_layers.search_strategy.progress import BeamGrowthStepEvent

                # For each candidate, find which layer was added by comparing with beam entries
                # Candidates are generated by adding one layer to a beam entry
                beam_layer_sets = [set(b["layers"]) for b in beam]
                tried = []
                for c in candidates:
                    candidate_set = set(c["layers"])
                    # Find the beam entry this candidate was expanded from
                    for beam_set in beam_layer_sets:
                        if beam_set < candidate_set and len(candidate_set - beam_set) == 1:
                            # Found the base, the added layer is the difference
                            added_layer = (candidate_set - beam_set).pop()
                            tried.append({"layer": added_layer, "score": c["score"]})
                            break

                # Sort by score descending for display
                tried.sort(key=lambda x: x["score"], reverse=True)

                # Base layers are from the best entry in previous beam
                base_layers = beam[0]["layers"] if beam else []

                self.observer.on_beam_growth_step(
                    BeamGrowthStepEvent(
                        step=growth_step + 1,  # +1 because step 1 was prescoring
                        total_steps=total_growth_steps,
                        phase_index=phase_index,
                        mixer_name=mixer_name,
                        base_layers=base_layers,
                        tried=tried,
                        beam=new_beam[:5],
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
            placement = LayerPlacement(assignments=assignments)
            candidate["score"] = await problem.evaluate(placement)
            self.eval_count += 1

            # Notify observer of eval completion
            if self.observer is not None:
                self.observer.on_eval_completed(self.eval_count)


class HierarchicalConfig(BaseCostConstrainedConfig):
    """Configuration for hierarchical beam search inner loop."""

    _impl_class: ClassVar[type["BaseSearchStrategy"]] = HierarchicalSearch

    strategy_type: Literal["cost_constrained_hierarchical"] = "cost_constrained_hierarchical"

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
            "Default 1 = greedy (lock in best after each phase). "
            "Set to beam_width to explore phase interactions at k× cost."
        ),
        gt=0,
    )
