"""Hierarchical beam search strategy implementation."""

import logging
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from place_layers.core.types import SearchResult
from place_layers.problem.base_problem import BaseProblem
from place_layers.problem.layer_placement import (
    LayerPlacement,
    LayerPlacementProblemConfig,
)
from place_layers.search_strategy.base_search_strategy import BaseSearchStrategy
from place_layers.search_strategy.base_search_strategy_config import (
    BaseSearchStrategyConfig,
)

logger = logging.getLogger(__name__)


class PhaseResult(BaseModel):
    """Results from a single search phase in hierarchical beam search."""

    phase_idx: int
    mixer_idx: int
    budget: int
    best_layers: list[int]
    best_score: float
    layer_scores: list[tuple[int, float]]


class HierarchicalBeamSearchResult(SearchResult[LayerPlacement]):
    """Search results with additional phase information for hierarchical beam search."""

    phase_results: list[PhaseResult]


class HierarchicalBeamSearch(BaseSearchStrategy[LayerPlacement]):
    """
    Hierarchical beam search for finding optimal mixer placement.

    The algorithm works in phases:
    1. Phase 0: Find best budgets[0] layers for mixer[0]
    2. Phase 1: Given fixed mixer[0] positions, find best budgets[1] layers for mixer[1]
    3. Continue for all budget phases
    4. Remaining layers use the last mixer
    """

    def __init__(self, config: "HierarchicalBeamSearchConfig") -> None:
        super().__init__(config)
        self.config: HierarchicalBeamSearchConfig  # Type narrowing
        self.eval_count = 0

    async def search(
        self,
        problem: BaseProblem[LayerPlacement],
    ) -> HierarchicalBeamSearchResult:
        """
        Run hierarchical beam search.

        Args:
            problem: Problem definition with embedded evaluator.

        Returns:
            Search results with best configuration.
        """
        # Extract problem parameters
        if not isinstance(problem.config, LayerPlacementProblemConfig):
            raise TypeError(
                f"HierarchicalBeamSearch requires LayerPlacementProblemConfig, "
                f"got {type(problem.config)}"
            )

        # Get dimensions from evaluator
        num_layers = problem.evaluator.config.num_layers
        num_mixers = problem.evaluator.config.num_mixers
        budgets = problem.config.budgets

        logger.info(f"\n{'='*60}")
        logger.info("Hierarchical Beam Search Configuration")
        logger.info(f"{'='*60}")
        logger.info(f"Total layers: {num_layers}")
        logger.info(f"Number of mixer types: {num_mixers}")
        logger.info(f"Budgets: {budgets}")
        logger.info(f"Beam width: {self.config.beam_width}")
        logger.info(f"Initial beam width: {self.config.initial_beam_width}")

        # Run phases
        fixed_assignments: dict[int, int] = {}
        phase_results: list[PhaseResult] = []

        for phase_idx, budget in enumerate(budgets):
            phase_result = await self._run_phase(
                phase_idx=phase_idx,
                budget=budget,
                num_layers=num_layers,
                fixed_assignments=fixed_assignments,
                problem=problem,
            )
            phase_results.append(phase_result)

            # Lock in the best layers from this phase
            for layer_idx in phase_result.best_layers:
                fixed_assignments[layer_idx] = phase_idx

        # Assign remaining layers to last mixer
        final_assignments = dict(fixed_assignments)
        for layer_idx in range(num_layers):
            if layer_idx not in final_assignments:
                final_assignments[layer_idx] = num_mixers - 1

        # Final evaluation
        assignments = tuple(final_assignments[i] for i in range(num_layers))
        final_config = LayerPlacement(assignments=assignments)
        final_score = await problem.evaluate(final_config)
        self.eval_count += 1

        logger.info(f"\n{'='*60}")
        logger.info("FINAL EVALUATION")
        logger.info(f"{'='*60}")
        logger.info("Final configuration:")
        for mixer_idx in range(num_mixers):
            layers = [
                l for l, m in final_assignments.items() if m == mixer_idx
            ]
            logger.info(
                f"  mixer[{mixer_idx}]: {len(layers)} layers - {sorted(layers)}"
            )
        logger.info(f"Final score: {final_score:.4f}")
        logger.info(f"Total evaluations: {self.eval_count}")

        return HierarchicalBeamSearchResult(
            best_parameters=final_config,
            best_score=final_score,
            total_evaluations=self.eval_count,
            phase_results=phase_results,
        )

    async def _run_phase(
        self,
        phase_idx: int,
        budget: int,
        num_layers: int,
        fixed_assignments: dict[int, int],
        problem: BaseProblem,
    ) -> PhaseResult:
        """Run one phase of hierarchical beam search."""
        mixer_idx = phase_idx
        fallback_mixer_idx = phase_idx + 1

        logger.info(f"\n{'='*60}")
        logger.info(
            f"PHASE {phase_idx + 1}: Optimizing placement for mixer[{mixer_idx}]"
        )
        logger.info(f"Budget: {budget} layers")
        logger.info(
            f"Baseline for non-assigned layers: mixer[{fallback_mixer_idx}]"
        )
        logger.info(f"{'='*60}")

        # Get available layers
        unassigned = [i for i in range(num_layers) if i not in fixed_assignments]
        logger.info(
            f"Unassigned layers: {len(unassigned)} out of {num_layers}"
        )

        # Pre-score individual layers
        layer_scores = await self._prescore_layers(
            unassigned=unassigned,
            mixer_idx=mixer_idx,
            fallback_mixer_idx=fallback_mixer_idx,
            num_layers=num_layers,
            fixed_assignments=fixed_assignments,
            problem=problem,
        )

        # Grow beam
        beam = await self._grow_beam(
            layer_scores=layer_scores,
            budget=budget,
            mixer_idx=mixer_idx,
            fallback_mixer_idx=fallback_mixer_idx,
            num_layers=num_layers,
            unassigned=unassigned,
            fixed_assignments=fixed_assignments,
            problem=problem,
        )

        best_candidate = beam[0]
        logger.info(f"\nPhase {phase_idx + 1} complete!")
        logger.info(
            f"Best layers for mixer[{mixer_idx}]: {best_candidate['layers']}"
        )
        logger.info(f"Best score: {best_candidate['score']:.4f}")

        return PhaseResult(
            phase_idx=phase_idx,
            mixer_idx=mixer_idx,
            budget=budget,
            best_layers=best_candidate["layers"],
            best_score=best_candidate["score"],
            layer_scores=layer_scores,
        )

    async def _prescore_layers(
        self,
        unassigned: list[int],
        mixer_idx: int,
        fallback_mixer_idx: int,
        num_layers: int,
        fixed_assignments: dict[int, int],
        problem: BaseProblem,
    ) -> list[tuple[int, float]]:
        """Pre-score individual layers to seed the beam."""
        logger.info("\nPre-scoring unassigned layers...")

        layer_scores = []
        for layer_idx in unassigned:
            assignment = self._create_assignment(
                fixed_assignments=fixed_assignments,
                target_layers=[layer_idx],
                target_mixer_idx=mixer_idx,
                unassigned=unassigned,
                fallback_mixer_idx=fallback_mixer_idx,
                num_layers=num_layers,
            )
            score = await problem.evaluate(assignment)
            self.eval_count += 1
            layer_scores.append((layer_idx, score))
            logger.debug(f"  Layer {layer_idx}: {score:.4f}")

        # Sort by score
        layer_scores.sort(
            key=lambda x: x[1], reverse=self.config.higher_is_better
        )

        logger.info(f"\nLayer ranking for mixer[{mixer_idx}]:")
        for rank, (layer_idx, score) in enumerate(layer_scores[:10]):
            logger.info(f"  {rank+1}. Layer {layer_idx}: {score:.4f}")

        return layer_scores

    async def _grow_beam(
        self,
        layer_scores: list[tuple[int, float]],
        budget: int,
        mixer_idx: int,
        fallback_mixer_idx: int,
        num_layers: int,
        unassigned: list[int],
        fixed_assignments: dict[int, int],
        problem: BaseProblem,
    ) -> list[dict]:
        """Grow the beam from seed to budget size."""
        logger.info(
            f"\nSeeding beam with top {self.config.initial_beam_width} layers..."
        )

        # Initialize beam with single-layer candidates
        beam = [
            {"layers": [layer_idx], "score": score}
            for layer_idx, score in layer_scores[: self.config.initial_beam_width]
        ]

        logger.info(f"\nGrowing beam to budget of {budget}...")
        best_score = beam[0]["score"]

        # Grow beam iteratively
        for growth_step in range(1, budget):
            logger.info(
                f"\nGrowth step {growth_step}: Adding layer #{growth_step+1}"
            )

            # Generate candidates
            candidates = self._generate_candidates(beam, unassigned)
            logger.info(f"Generated {len(candidates)} unique candidates")

            # Evaluate candidates
            await self._evaluate_candidates(
                candidates=candidates,
                mixer_idx=mixer_idx,
                fallback_mixer_idx=fallback_mixer_idx,
                num_layers=num_layers,
                unassigned=unassigned,
                fixed_assignments=fixed_assignments,
                problem=problem,
            )

            # Select top candidates for new beam
            candidates.sort(
                key=lambda x: x["score"], reverse=self.config.higher_is_better
            )
            beam = candidates[: self.config.beam_width]

            self._log_top_candidates(beam)

            # Check early stopping
            new_best_score = beam[0]["score"]
            if self._should_early_stop(best_score, new_best_score):
                break
            best_score = new_best_score

        return beam

    def _generate_candidates(
        self, beam: list[dict], unassigned: list[int]
    ) -> list[dict]:
        """Generate new candidates by expanding each beam entry."""
        candidates = []
        seen: set[tuple[int, ...]] = set()

        for beam_entry in beam:
            existing = set(beam_entry["layers"])

            for layer_idx in unassigned:
                if layer_idx in existing:
                    continue

                new_layers = tuple(sorted(beam_entry["layers"] + [layer_idx]))

                if new_layers in seen:
                    continue
                seen.add(new_layers)

                candidates.append({"layers": list(new_layers), "score": None})

        return candidates

    async def _evaluate_candidates(
        self,
        candidates: list[dict],
        mixer_idx: int,
        fallback_mixer_idx: int,
        num_layers: int,
        unassigned: list[int],
        fixed_assignments: dict[int, int],
        problem: BaseProblem,
    ) -> None:
        """Evaluate all candidates and store scores."""
        for i, candidate in enumerate(candidates):
            assignment = self._create_assignment(
                fixed_assignments=fixed_assignments,
                target_layers=candidate["layers"],
                target_mixer_idx=mixer_idx,
                unassigned=unassigned,
                fallback_mixer_idx=fallback_mixer_idx,
                num_layers=num_layers,
            )
            candidate["score"] = await problem.evaluate(assignment)
            self.eval_count += 1

            if (i + 1) % max(1, len(candidates) // 10) == 0:
                logger.debug(f"  Evaluated {i+1}/{len(candidates)} candidates...")

    def _create_assignment(
        self,
        fixed_assignments: dict[int, int],
        target_layers: list[int],
        target_mixer_idx: int,
        unassigned: list[int],
        fallback_mixer_idx: int,
        num_layers: int,
    ) -> LayerPlacement:
        """Create a complete assignment for evaluation."""
        assignment = dict(fixed_assignments)

        # Assign target layers to target mixer
        for layer_idx in target_layers:
            assignment[layer_idx] = target_mixer_idx

        # Assign unassigned layers to fallback
        for layer_idx in unassigned:
            if layer_idx not in assignment:
                assignment[layer_idx] = fallback_mixer_idx

        # Convert dict to tuple
        assignments = tuple(assignment[i] for i in range(num_layers))
        return LayerPlacement(assignments=assignments)

    def _log_top_candidates(self, beam: list[dict]) -> None:
        """Log the top candidates in the beam."""
        logger.info(f"\nTop {min(3, len(beam))} candidates:")
        for i, candidate in enumerate(beam[:3]):
            logger.info(
                f"  {i+1}. {candidate['layers']} - Score: {candidate['score']:.4f}"
            )

    def _should_early_stop(self, best_score: float, new_best_score: float) -> bool:
        """Check if early stopping criteria is met."""
        if self.config.higher_is_better:
            improvement = new_best_score - best_score
        else:
            improvement = best_score - new_best_score

        if improvement < self.config.early_stop_threshold:
            logger.info(
                f"Early stopping: improvement {improvement:.4f} < "
                f"threshold {self.config.early_stop_threshold}"
            )
            return True
        return False


class HierarchicalBeamSearchConfig(BaseSearchStrategyConfig):
    """Configuration for hierarchical beam search."""

    _impl_class: ClassVar[type[BaseSearchStrategy]] = HierarchicalBeamSearch

    strategy_type: Literal["hierarchical_beam"] = "hierarchical_beam"

    beam_width: int = Field(
        default=12,
        description="Number of top candidates to keep at each growth step",
        gt=0,
    )

    initial_beam_width: int = Field(
        default=12,
        description="Number of top single-layer configs to seed each phase",
        gt=0,
    )

    early_stop_threshold: float = Field(
        default=0.001,
        description="Stop if best score improvement is below this threshold",
        ge=0.0,
    )

    higher_is_better: bool = Field(
        default=True,
        description="Whether higher metric values are better",
    )
