"""Hierarchical beam search for cost-constrained optimization.

This implements an inner loop strategy that uses the hierarchical beam search
approach: phase-by-phase assignment of layers to mixers.
"""

from __future__ import annotations

import logging
from math import comb
from typing import TYPE_CHECKING, ClassVar, Literal, Sequence

from pydantic import Field

from place_layers.core.types import LayerPlacement
from place_layers.cost_model.base_cost_model import Allocation, BaseCostModel
from place_layers.problem.layer_placement import CostConstrainedProblem
from place_layers.search_strategy.beam_core import (
    create_layer_assignment,
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
# Evaluation count calculation for progress estimation
# -----------------------------------------------------------------------------


def phase_eval_count(
    budget: int, unassigned: int, beam_width: int, is_first_phase: bool
) -> int:
    """Expected evaluation count for a phase.

    Args:
        budget: Number of layers to assign in this phase.
        unassigned: Number of unassigned layers at phase start.
        beam_width: Beam width parameter (B).
        is_first_phase: Whether this is the first phase (single beam entry input).

    Returns:
        Expected number of evaluations for this phase.
    """
    if budget <= 0:
        return 0

    # Prescore: for first phase there's 1 beam entry, otherwise B entries
    # Each entry prescores all its unassigned layers
    if is_first_phase:
        prescore_count = unassigned
    else:
        prescore_count = beam_width * unassigned

    if budget == 1:
        # Prescore only
        return prescore_count

    # Growth steps
    growth_count = 0
    for s in range(1, budget):
        # Each growth step: B parents * (unassigned - s) candidates
        # But capped by C(unassigned, s+1) unique combinations (deduplication)
        growth_count += min(beam_width * (unassigned - s), comb(unassigned, s + 1))

    return prescore_count + growth_count


def allocation_eval_count(
    budgets: list[int],
    num_layers: int,
    beam_width: int,
) -> int:
    """Calculate expected evaluation count for a single allocation.

    Aggregates phase counts across all phases.

    Args:
        budgets: Phase budgets (layers to assign in each phase, excluding last).
        num_layers: Total number of layers (N).
        beam_width: Beam width parameter (B).

    Returns:
        Expected number of evaluations for this allocation.
    """
    # Final evaluation always happens
    count = 1
    unassigned = num_layers
    is_first = True

    for budget in budgets:
        if budget == 0:
            continue

        count += phase_eval_count(budget, unassigned, beam_width, is_first)
        unassigned -= budget
        is_first = False

    return count


class HierarchicalSearch(BaseCostConstrainedSearch):
    """
    Cost-constrained search using hierarchical beam search for the inner loop.

    Within each allocation, uses the hierarchical approach with unified beam:
    - Phase 1: Find best layers for mixer 0 (e.g., FA)
    - Phase 2: Given fixed mixer 0 positions, find best layers for mixer 1 (e.g., SWA)
    - Phase 3: Remaining layers get mixer 2 (e.g., GDN)

    Each beam entry carries its own fixed_assignments from prior phases, allowing
    different "paths" to emerge naturally through competition. Entries with weak
    histories get eliminated as the beam evolves.

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
        observers: "Sequence[SearchObserver]" = (),
        progress_estimator: "SearchProgressEstimator | None" = None,
    ) -> None:
        super().__init__(config, observers=observers, progress_estimator=progress_estimator)
        self.config: HierarchicalConfig

    def _build_partial_placement(
        self,
        fixed: dict[int, str],
        num_layers: int,
    ) -> tuple[str | None, ...]:
        """Build a partial placement from fixed assignments.

        Args:
            fixed: Dict mapping layer index to mixer name for assigned layers.
            num_layers: Total number of layers.

        Returns:
            Tuple where index=layer, value=mixer or None if unassigned.
        """
        return tuple(fixed.get(i) for i in range(num_layers))

    def _build_partial_placement_with_current(
        self,
        fixed: dict[int, str],
        current_layers: list[int],
        current_mixer: str,
        num_layers: int,
    ) -> tuple[str | None, ...]:
        """Build a partial placement including current phase assignments.

        Args:
            fixed: Dict mapping layer index to mixer name for prior phases.
            current_layers: Layers assigned in current phase.
            current_mixer: Mixer being assigned in current phase.
            num_layers: Total number of layers.

        Returns:
            Tuple where index=layer, value=mixer or None if unassigned.
        """
        result: list[str | None] = [None] * num_layers
        for layer_idx, mixer in fixed.items():
            result[layer_idx] = mixer
        for layer_idx in current_layers:
            result[layer_idx] = current_mixer
        return tuple(result)

    async def _search_within_allocation(
        self,
        problem: CostConstrainedProblem,
        cost_model: BaseCostModel,
        allocation: Allocation,
        num_layers: int,
    ) -> BeamEntry | None:
        """
        Search for the best assignment within a fixed allocation using unified beam search.

        Uses a single unified beam throughout all phases. Each beam entry carries its own
        fixed_assignments from prior phases, so different "paths" emerge naturally through
        competition. Entries compete at every step and across phases - weak entries get
        eliminated regardless of their history.

        Phase order is determined by the order of keys in mixer_costs config, not alphabetical.
        """
        alloc_cost = cost_model.allocation_cost(allocation)
        mixer_names = problem.evaluator.mixer_names  # Canonical order for allocations
        phase_order = cost_model.phase_order  # User-specified order for phases

        # Build mapping from mixer name to allocation index
        name_to_alloc_idx = {name: i for i, name in enumerate(mixer_names)}

        # Start with a single beam entry (no fixed assignments yet)
        beam: list[dict] = [
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

            # Emit phase started event (running_phase_idx is 0-based)
            if self.observers:
                from place_layers.search_strategy.progress import PhaseStartedEvent

                event = PhaseStartedEvent(
                    phase_index=running_phase_idx,
                    total_phases=total_phases,
                    mixer_name=mixer_name,
                    budget=budget,
                )
                for observer in self.observers:
                    observer.on_phase_started(event)

            # Run phase - returns updated beam entries with new fixed assignments
            beam = await self._run_phase(
                problem=problem,
                beam=beam,
                budget=budget,
                mixer_name=mixer_name,
                fallback_mixer=fallback_mixer,
                num_layers=num_layers,
                phase_index=running_phase_idx,
            )

            if not beam:
                # No valid entries remain
                return None

            # Emit phase completed event
            if self.observers:
                from place_layers.search_strategy.progress import PhaseCompletedEvent

                # Get the layers assigned in this phase from the best entry
                best_entry = beam[0]
                phase_layers = sorted(
                    [
                        layer_idx
                        for layer_idx, mixer in best_entry["fixed"].items()
                        if mixer == mixer_name
                    ]
                )

                event = PhaseCompletedEvent(
                    phase_index=running_phase_idx,
                    total_phases=total_phases,
                    mixer_name=mixer_name,
                    budget=budget,
                    final_layers=tuple(phase_layers),
                    final_score=best_entry["score"],
                )
                for observer in self.observers:
                    observer.on_phase_completed(event)

            # Increment after phase completes (0-based indexing)
            running_phase_idx += 1

        # Take best entry for final assignment
        best_entry = beam[0]

        # Assign remaining layers to the last mixer (last in phase_order)
        last_mixer = phase_order[-1]
        for layer_idx in best_entry["unassigned"]:
            best_entry["fixed"][layer_idx] = last_mixer

        # Build final assignment list
        assignments = [best_entry["fixed"][i] for i in range(num_layers)]

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
        beam: list[dict],
        budget: int,
        mixer_name: str,
        fallback_mixer: str,
        num_layers: int,
        phase_index: int,
    ) -> list[dict]:
        """
        Run one phase with unified beam: find best 'budget' layers for mixer_name.

        Each beam entry carries its own fixed/unassigned state from prior phases.
        The phase prescores all unassigned layers for each entry, then grows the
        beam to budget size. Finally, updates fixed_assignments for each entry.

        Args:
            problem: The optimization problem.
            beam: Current beam entries, each with "fixed", "unassigned", "score" keys.
            budget: Number of layers to assign to mixer_name in this phase.
            mixer_name: The mixer type to assign layers to.
            fallback_mixer: Mixer for remaining unassigned layers during evaluation.
            num_layers: Total number of layers.
            phase_index: Current phase index for progress reporting.

        Returns:
            Updated beam entries with new fixed_assignments for this phase.
        """
        # Emit step started event before prescoring
        if self.observers:
            from place_layers.search_strategy.progress import BeamGrowthStepStartedEvent
            from place_layers.search_strategy.progress.events import BeamEntry as EventBeamEntry

            total_candidates = sum(len(e["unassigned"]) for e in beam)

            # Build partial placements for parents
            parent_entries = tuple(
                EventBeamEntry(
                    partial_placement=self._build_partial_placement(
                        e.get("fixed", {}), num_layers
                    ),
                    score=e.get("score"),
                )
                for e in beam
            )

            event = BeamGrowthStepStartedEvent(
                step=1,
                total_steps=budget,
                phase_index=phase_index,
                mixer_name=mixer_name,
                num_candidates=total_candidates,
                parents=parent_entries,
            )
            for observer in self.observers:
                observer.on_beam_growth_step_started(event)

        # Pre-score individual layers for each beam entry
        # Creates candidates: each is a single-layer entry inheriting parent's state
        prescored: list[dict] = []
        # Track all tried candidates with parent_idx for event reporting
        all_tried: list[dict] = []

        for parent_idx, entry in enumerate(beam):
            if not entry["unassigned"]:
                continue

            layer_scores = await self._prescore_layers(
                problem=problem,
                unassigned=entry["unassigned"],
                mixer_name=mixer_name,
                fallback_mixer=fallback_mixer,
                fixed_assignments=entry["fixed"],
                num_layers=num_layers,
                parent_idx=parent_idx,
                step=1,
                total_steps=budget,
                phase_index=phase_index,
            )

            # Track all layer scores with parent_idx and placement for event reporting
            for layer_idx, score, placement in layer_scores:
                all_tried.append({
                    "layer": layer_idx,
                    "score": score,
                    "parent_idx": parent_idx,
                    "placement": placement,
                })

            # Create single-layer candidates from prescoring
            for layer_idx, score, placement in layer_scores:
                prescored.append({
                    "fixed": entry["fixed"],
                    "unassigned": entry["unassigned"],
                    "layers": [layer_idx],
                    "score": score,
                    "parent_idx": parent_idx,
                    "added_layer": layer_idx,
                    "placement": placement,
                })

        if not prescored:
            return []

        # Select top B entries from prescoring
        prescored.sort(key=lambda x: x["score"], reverse=True)
        working_beam = prescored[: self.config.beam_width]

        # Emit step 1 event for initial beam (from prescoring)
        if self.observers:
            from place_layers.search_strategy.progress import BeamGrowthStepEvent
            from place_layers.search_strategy.progress.events import (
                BeamEntry as EventBeamEntry,
                EvaluatedCandidate,
                SelectedEntry,
            )

            # Build parents from input beam
            parents = tuple(
                EventBeamEntry(
                    partial_placement=self._build_partial_placement(
                        e.get("fixed", {}), num_layers
                    ),
                    score=e.get("score"),
                )
                for e in beam
            )

            # Build tried candidates (sorted by score descending)
            sorted_all_tried = sorted(all_tried, key=lambda x: x["score"], reverse=True)
            tried = tuple(
                EvaluatedCandidate(
                    layer=t["layer"],
                    score=t["score"],
                    parent_idx=t["parent_idx"],
                    placement=t["placement"],
                )
                for t in sorted_all_tried
            )

            # Build beam entries
            beam_entries = tuple(
                SelectedEntry(
                    partial_placement=self._build_partial_placement_with_current(
                        b.get("fixed", {}), b["layers"], mixer_name, num_layers
                    ),
                    score=b["score"],
                    parent_idx=b["parent_idx"],
                    added_layer=b["added_layer"],
                )
                for b in working_beam
            )

            # Compute best_candidate and cut_candidates for observers
            best_candidate = tried[0] if tried else None
            beam_keys = {(b["parent_idx"], b["added_layer"]) for b in working_beam}
            cut_candidates = tuple(
                t for t in tried if (t.parent_idx, t.layer) not in beam_keys
            )

            event = BeamGrowthStepEvent(
                step=1,
                total_steps=budget,
                phase_index=phase_index,
                mixer_name=mixer_name,
                parents=parents,
                tried=tried,
                beam=beam_entries,
                best_candidate=best_candidate,
                cut_candidates=cut_candidates,
            )
            for observer in self.observers:
                observer.on_beam_growth_step(event)

        if budget == 1:
            # Finalize phase: update fixed assignments
            return self._finalize_phase(working_beam, mixer_name)

        # Grow beam to budget size
        working_beam = await self._grow_beam(
            problem=problem,
            working_beam=working_beam,
            budget=budget,
            mixer_name=mixer_name,
            fallback_mixer=fallback_mixer,
            num_layers=num_layers,
            phase_index=phase_index,
        )

        # Finalize phase: update fixed assignments
        return self._finalize_phase(working_beam, mixer_name)

    def _finalize_phase(self, beam: list[dict], mixer_name: str) -> list[dict]:
        """Update fixed assignments for phase completion.

        Takes working beam entries with "layers" (layers assigned in this phase)
        and updates their fixed/unassigned state to reflect the new assignments.
        """
        result = []
        for entry in beam:
            new_fixed = dict(entry["fixed"])
            new_unassigned = list(entry["unassigned"])
            for layer_idx in entry["layers"]:
                new_fixed[layer_idx] = mixer_name
                new_unassigned.remove(layer_idx)
            result.append({
                "fixed": new_fixed,
                "unassigned": new_unassigned,
                "score": entry["score"],
            })
        return result

    async def _prescore_layers(
        self,
        problem: CostConstrainedProblem,
        unassigned: list[int],
        mixer_name: str,
        fallback_mixer: str,
        fixed_assignments: dict[int, str],
        num_layers: int,
        parent_idx: int = 0,
        step: int = 1,
        total_steps: int = 1,
        phase_index: int = 1,
    ) -> list[tuple[int, float, tuple[str, ...]]]:
        """Pre-score individual layers to seed the beam.

        Returns:
            List of (layer_idx, score, complete_placement) tuples, sorted by score.
        """
        layer_scores: list[tuple[int, float, tuple[str, ...]]] = []
        total_evals = len(unassigned)

        for eval_idx, layer_idx in enumerate(unassigned, 1):
            assignments = create_layer_assignment(
                fixed_assignments=fixed_assignments,
                target_layers=[layer_idx],
                target_mixer=mixer_name,
                unassigned=unassigned,
                fallback_mixer=fallback_mixer,
                num_layers=num_layers,
            )
            complete_placement = tuple(assignments)
            placement = LayerPlacement(assignments=complete_placement)
            score = await problem.evaluate(placement)
            self.eval_count += 1

            # Notify observers of eval completion
            if self.observers:
                for observer in self.observers:
                    observer.on_eval_completed(self.eval_count)

                # Emit candidate evaluated event
                from place_layers.search_strategy.progress import CandidateEvaluatedEvent

                event = CandidateEvaluatedEvent(
                    step=step,
                    total_steps=total_steps,
                    phase_index=phase_index,
                    mixer_name=mixer_name,
                    layer=layer_idx,
                    score=score,
                    parent_idx=parent_idx,
                    placement=complete_placement,
                    eval_index=eval_idx,
                    total_evals=total_evals,
                )
                for observer in self.observers:
                    observer.on_candidate_evaluated(event)

            layer_scores.append((layer_idx, score, complete_placement))

        # Sort by score (higher is better)
        layer_scores.sort(key=lambda x: x[1], reverse=True)
        return layer_scores

    async def _grow_beam(
        self,
        problem: CostConstrainedProblem,
        working_beam: list[dict],
        budget: int,
        mixer_name: str,
        fallback_mixer: str,
        num_layers: int,
        phase_index: int,
    ) -> list[dict]:
        """Grow the beam from prescored single-layer entries to budget size.

        Each beam entry carries its own fixed/unassigned state from prior phases.
        Candidates are generated by expanding each entry with layers from its
        own unassigned list.
        """
        # Grow beam iteratively (step 1 was prescoring, handled in _run_phase)
        for growth_step in range(1, budget):
            # Generate candidates - each entry uses its own unassigned list
            candidates = self._generate_candidates_unified(working_beam)

            if not candidates:
                break

            # Emit step started event before evaluating
            if self.observers:
                from place_layers.search_strategy.progress import BeamGrowthStepStartedEvent
                from place_layers.search_strategy.progress.events import (
                    BeamEntry as EventBeamEntry,
                )

                parent_entries = tuple(
                    EventBeamEntry(
                        partial_placement=self._build_partial_placement_with_current(
                            b.get("fixed", {}), b["layers"], mixer_name, num_layers
                        ),
                        score=b["score"],
                    )
                    for b in working_beam
                )

                event = BeamGrowthStepStartedEvent(
                    step=growth_step + 1,  # +1 because step 1 was prescoring
                    total_steps=budget,
                    phase_index=phase_index,
                    mixer_name=mixer_name,
                    num_candidates=len(candidates),
                    parents=parent_entries,
                )
                for observer in self.observers:
                    observer.on_beam_growth_step_started(event)

            # Evaluate candidates
            await self._evaluate_candidates_unified(
                problem=problem,
                candidates=candidates,
                mixer_name=mixer_name,
                fallback_mixer=fallback_mixer,
                num_layers=num_layers,
                step=growth_step + 1,
                total_steps=budget,
                phase_index=phase_index,
            )

            # Select top candidates using shared function (higher is better)
            new_beam = select_top_beam(
                candidates, self.config.beam_width, higher_is_better=True
            )

            # Emit beam growth step event with expansion details
            if self.observers:
                from place_layers.search_strategy.progress import BeamGrowthStepEvent
                from place_layers.search_strategy.progress.events import (
                    BeamEntry as EventBeamEntry,
                    EvaluatedCandidate,
                    SelectedEntry,
                )

                # Build tried list with provenance from candidates
                # Sort by score descending for display
                sorted_candidates = sorted(
                    candidates, key=lambda x: x["score"], reverse=True
                )
                tried = tuple(
                    EvaluatedCandidate(
                        layer=c["added_layer"],
                        score=c["score"],
                        parent_idx=c["parent_idx"],
                        placement=c["placement"],
                    )
                    for c in sorted_candidates
                )

                # Build parents list from previous beam
                parents = tuple(
                    EventBeamEntry(
                        partial_placement=self._build_partial_placement_with_current(
                            b.get("fixed", {}), b["layers"], mixer_name, num_layers
                        ),
                        score=b["score"],
                    )
                    for b in working_beam
                )

                # Build beam with provenance
                beam_entries = tuple(
                    SelectedEntry(
                        partial_placement=self._build_partial_placement_with_current(
                            b.get("fixed", {}), b["layers"], mixer_name, num_layers
                        ),
                        score=b["score"],
                        parent_idx=b["parent_idx"],
                        added_layer=b["added_layer"],
                    )
                    for b in new_beam
                )

                # Compute best_candidate and cut_candidates for observers
                best_candidate = tried[0] if tried else None
                beam_keys = {(b["parent_idx"], b["added_layer"]) for b in new_beam}
                cut_candidates = tuple(
                    t for t in tried if (t.parent_idx, t.layer) not in beam_keys
                )

                event = BeamGrowthStepEvent(
                    step=growth_step + 1,  # +1 because step 1 was prescoring
                    total_steps=budget,
                    phase_index=phase_index,
                    mixer_name=mixer_name,
                    parents=parents,
                    tried=tried,
                    beam=beam_entries,
                    best_candidate=best_candidate,
                    cut_candidates=cut_candidates,
                )
                for observer in self.observers:
                    observer.on_beam_growth_step(event)

            working_beam = new_beam

        return working_beam

    def _generate_candidates_unified(self, beam: list[dict]) -> list[dict]:
        """Generate candidates for unified beam where each entry has its own unassigned list.

        For each entry, tries adding each unassigned layer that isn't already in its layers.
        Candidates inherit the parent's fixed/unassigned state.
        """
        candidates = []
        seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

        for parent_idx, entry in enumerate(beam):
            existing = set(entry["layers"])
            # Use the entry's own unassigned list
            parent_unassigned = entry["unassigned"]

            for layer_idx in parent_unassigned:
                if layer_idx in existing:
                    continue

                new_layers = tuple(sorted(entry["layers"] + [layer_idx]))

                # Deduplicate based on (fixed_key, layers) - entries with same fixed
                # context and same layers are duplicates
                fixed_key = tuple(sorted(entry["fixed"].items()))
                dedup_key = (fixed_key, new_layers)

                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                candidates.append({
                    "fixed": entry["fixed"],
                    "unassigned": parent_unassigned,
                    "layers": list(new_layers),
                    "score": None,
                    "parent_idx": parent_idx,
                    "added_layer": layer_idx,
                })

        return candidates

    async def _evaluate_candidates_unified(
        self,
        problem: CostConstrainedProblem,
        candidates: list[dict],
        mixer_name: str,
        fallback_mixer: str,
        num_layers: int,
        step: int = 1,
        total_steps: int = 1,
        phase_index: int = 1,
    ) -> None:
        """Evaluate candidates where each has its own fixed/unassigned context."""
        total_evals = len(candidates)
        for eval_idx, candidate in enumerate(candidates, 1):
            assignments = create_layer_assignment(
                fixed_assignments=candidate["fixed"],
                target_layers=candidate["layers"],
                target_mixer=mixer_name,
                unassigned=candidate["unassigned"],
                fallback_mixer=fallback_mixer,
                num_layers=num_layers,
            )
            complete_placement = tuple(assignments)
            placement = LayerPlacement(assignments=complete_placement)
            candidate["score"] = await problem.evaluate(placement)
            candidate["placement"] = complete_placement  # Store for event reporting
            self.eval_count += 1

            # Notify observers of eval completion
            if self.observers:
                for observer in self.observers:
                    observer.on_eval_completed(self.eval_count)

                # Emit candidate evaluated event
                from place_layers.search_strategy.progress import CandidateEvaluatedEvent

                event = CandidateEvaluatedEvent(
                    step=step,
                    total_steps=total_steps,
                    phase_index=phase_index,
                    mixer_name=mixer_name,
                    layer=candidate["added_layer"],
                    score=candidate["score"],
                    parent_idx=candidate["parent_idx"],
                    placement=complete_placement,
                    eval_index=eval_idx,
                    total_evals=total_evals,
                )
                for observer in self.observers:
                    observer.on_candidate_evaluated(event)


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

    def create_progress_estimator(
        self, problem: "BaseProblem"
    ) -> "SearchProgressEstimator | None":
        """Create progress estimator for hierarchical beam search.

        Uses expected evaluation counts with variance bounds to account for
        deduplication effects during beam growth (which depends on which
        specific layers are selected at each step).

        Args:
            problem: The optimization problem (must be CostConstrainedProblem).

        Returns:
            Configured SearchProgressEstimator, or None if not applicable.
        """
        import math

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

        # Model uncertainty coefficient of variation (~15%)
        # Accounts for deduplication variance in growth steps and
        # multi-phase beam carryover effects
        model_cv = 0.15

        # Compute counts for each allocation
        total_expected = 0.0
        total_variance = 0.0
        allocation_info: list[AllocationProgressInfo] = []

        for alloc_cost, allocation in allocations:
            # Get phase budgets in phase_order (excluding last which gets remainder)
            budgets = tuple(
                allocation[name_to_alloc_idx[phase_order[i]]]
                for i in range(len(phase_order) - 1)
            )

            # Compute expected evaluation count
            alloc_count = allocation_eval_count(
                budgets=list(budgets),
                num_layers=num_layers,
                beam_width=self.beam_width,
            )

            # Add model uncertainty variance
            alloc_variance = (model_cv * alloc_count) ** 2
            alloc_std = math.sqrt(alloc_variance)

            total_expected += alloc_count
            total_variance += alloc_variance

            allocation_info.append(
                AllocationProgressInfo(
                    allocation=allocation,
                    budgets=budgets,
                    cost=alloc_cost,
                    expected=float(alloc_count),
                    variance=alloc_variance,
                    estimated_lower=int(alloc_count - 2 * alloc_std),
                    estimated_upper=int(alloc_count + 2 * alloc_std),
                )
            )

        # Compute total bounds (95% CI)
        total_std = math.sqrt(total_variance)

        return SearchProgressEstimator(
            initial=ProgressEstimate(
                lower=int(total_expected - 2 * total_std),
                upper=int(total_expected + 2 * total_std),
                expected=int(total_expected),
                std_dev=total_std,
            ),
            allocation_info=tuple(allocation_info),
        )
