"""Surrogate-assisted search strategy for layer placement optimization.

Uses a Ridge + Bootstrap ensemble surrogate trained on a pre-built
exploration dataset to guide evaluation toward promising placements via
Expected Improvement acquisition.

The surrogate is a DRAFTER: it proposes candidates cheaply,
but the real evaluator (via problem.evaluate) validates them.

Requires an exploration dataset (from explore_placements command).
Refinement: surrogate drafts candidates -> EI selects top-k -> real evaluator validates.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
from pydantic import Field
from scipy.stats import spearmanr

from place_layers.annealing.sa_core import SAConfig, run_multi_start_sa
from place_layers.core.types import LayerPlacement
from place_layers.cost_model.base_cost_model import BaseCostModel
from place_layers.problem.layer_placement import CostConstrainedProblem
from place_layers.search_strategy.cost_constrained.base import BaseCostConstrainedSearch
from place_layers.search_strategy.cost_constrained.base_config import (
    BaseCostConstrainedConfig,
)
from place_layers.search_strategy.cost_constrained.types import (
    BeamEntry,
    CostConstrainedSearchResult,
)
from place_layers.cost_model.enumeration import enumerate_feasible_allocations
from place_layers.surrogate.acquisition import expected_improvement
from place_layers.surrogate.dataset import ExplorationDataset
from place_layers.surrogate.ridge_surrogate import RidgeSurrogate

if TYPE_CHECKING:
    from place_layers.cost_model.base_cost_model import Allocation
    from place_layers.search_strategy.base_search_strategy import BaseSearchStrategy
    from place_layers.search_strategy.progress import SearchObserver
    from place_layers.search_strategy.progress.estimator import SearchProgressEstimator


logger = logging.getLogger(__name__)


# ============================================================================
# Strategy
# ============================================================================


class SurrogateSearch(BaseCostConstrainedSearch):
    """Surrogate-assisted search using Ridge + Bootstrap ensemble + EI.

    Searches across the full placement space simultaneously (no per-allocation
    inner loop). Requires a pre-built exploration dataset. Refinement uses
    EI-guided selection of SA-generated candidates.
    """

    config: "SurrogateSearchConfig"

    async def _search_max_performance(
        self,
        problem: CostConstrainedProblem,
        cost_model: BaseCostModel,
        num_layers: int,
        cost_budget: float,
        cost_floor: float | None = None,
    ) -> CostConstrainedSearchResult:
        """MAX_PERFORMANCE mode: maximize performance subject to cost <= budget."""
        logger.debug("Mode: MAX_PERFORMANCE (surrogate-assisted), budget=%.4f", cost_budget)

        # Enumerate feasible allocations within budget
        feasible_allocations = enumerate_feasible_allocations(
            cost_model, num_layers, cost_budget, cost_floor=cost_floor
        )
        logger.debug("Found %d feasible allocations", len(feasible_allocations))

        if not feasible_allocations:
            raise ValueError("No feasible allocations found within budget")

        # Run refinement loop
        all_assignments, all_scores = await self._run_surrogate_loop(
            problem=problem,
            cost_model=cost_model,
            num_layers=num_layers,
            feasible_allocations=feasible_allocations,
            cost_budget=cost_budget,
            cost_floor=cost_floor,
        )

        # Select best only from within-budget placements (exploration may be unconstrained)
        costs = [
            cost_model.allocation_cost(dict(Counter(a))) for a in all_assignments
        ]
        feasible_indices = [i for i, c in enumerate(costs) if c <= cost_budget]
        if not feasible_indices:
            raise ValueError("No feasible solutions found within budget")

        feasible_scores = [all_scores[i] for i in feasible_indices]
        best_feasible = feasible_indices[int(np.argmax(feasible_scores))]
        best_assignment = all_assignments[best_feasible]
        best_alloc = dict(Counter(best_assignment))
        best_score = all_scores[best_feasible]
        best_entry = BeamEntry(
            assignments=best_assignment,
            allocation=best_alloc,
            score=best_score,
        )

        return self._create_result(
            best_entry=best_entry,
            cost_model=cost_model,
            mode="max_performance",
            constraint_value=cost_budget,
            allocations_explored=0,
        )

    async def _search_min_cost(
        self,
        problem: CostConstrainedProblem,
        cost_model: BaseCostModel,
        num_layers: int,
        performance_target: float,
        cost_floor: float | None = None,
    ) -> CostConstrainedSearchResult:
        """MIN_COST mode: minimize cost subject to performance >= target."""
        logger.debug("Mode: MIN_COST (surrogate-assisted), target=%.4f", performance_target)

        # Enumerate ALL allocations (no budget filter for min_cost)
        feasible_allocations = enumerate_feasible_allocations(
            cost_model, num_layers, cost_budget=None, cost_floor=cost_floor
        )
        logger.debug("Found %d total allocations", len(feasible_allocations))

        if not feasible_allocations:
            raise ValueError("No allocations found")

        # Run refinement loop
        all_assignments, all_scores = await self._run_surrogate_loop(
            problem=problem,
            cost_model=cost_model,
            num_layers=num_layers,
            feasible_allocations=feasible_allocations,
            cost_budget=None,
            cost_floor=cost_floor,
        )

        # Find cheapest that meets target
        best_assignment: tuple[str, ...] | None = None
        best_alloc: dict[str, int] | None = None
        best_cost = float("inf")
        # Also track best by score as fallback
        fallback_assignment: tuple[str, ...] | None = None
        fallback_alloc: dict[str, int] | None = None
        fallback_score = float("-inf")

        for assignment, score in zip(all_assignments, all_scores):
            alloc = dict(Counter(assignment))
            cost = cost_model.allocation_cost(alloc)

            if score >= performance_target and cost < best_cost:
                best_cost = cost
                best_assignment = assignment
                best_alloc = alloc
            if score > fallback_score:
                fallback_score = score
                fallback_assignment = assignment
                fallback_alloc = alloc

        if best_assignment is None:
            logger.warning(
                "No placement met performance target %s. "
                "Returning best by score.",
                performance_target,
            )
            assert fallback_assignment is not None and fallback_alloc is not None
            best_assignment = fallback_assignment
            best_alloc = fallback_alloc

        assert best_alloc is not None
        # Get score for best assignment
        placement = LayerPlacement(assignments=best_assignment)
        best_score_val = await problem.evaluate(placement)
        self.eval_count += 1
        best_entry = BeamEntry(
            assignments=best_assignment,
            allocation=best_alloc,
            score=best_score_val,
        )

        return self._create_result(
            best_entry=best_entry,
            cost_model=cost_model,
            mode="min_cost",
            constraint_value=performance_target,
            allocations_explored=0,
        )

    async def _run_surrogate_loop(
        self,
        problem: CostConstrainedProblem,
        cost_model: BaseCostModel,
        num_layers: int,
        feasible_allocations: list["Allocation"],
        cost_budget: float | None,
        cost_floor: float | None = None,
    ) -> tuple[list[tuple[str, ...]], list[float]]:
        """Load exploration data and run refinement, return (all_assignments, all_scores)."""
        rng = random.Random(self.config.sa.seed)
        mixer_names = cost_model.phase_order

        # Load exploration dataset
        exploration = ExplorationDataset.load(
            Path(self.config.exploration_dataset_path)
        )
        all_assignments: list[tuple[str, ...]] = list(exploration.assignments)
        all_scores: list[float] = list(exploration.scores)
        logger.info(
            "Loaded exploration dataset: %d samples, rho=%.4f",
            len(all_assignments),
            exploration.final_rho,
        )

        # Emit dataset loaded event
        if self.observers:
            from place_layers.search_strategy.progress import SurrogateDatasetLoadedEvent

            dataset_event = SurrogateDatasetLoadedEvent(
                num_samples=len(all_assignments),
                exploration_rho=exploration.final_rho,
            )
            for observer in self.observers:
                observer.on_surrogate_dataset_loaded(dataset_event)

        # Refinement via SA against surrogate
        logger.debug("Refinement: %d rounds", self.config.num_rounds)

        surrogate = RidgeSurrogate(
            mixer_names=mixer_names,
            seed=self.config.sa.seed,
            feature_config=exploration.feature_config,
        )
        surrogate.fit(all_assignments, np.array(all_scores))

        # f_best: only from within-budget placements (exploration may be unconstrained)
        if cost_budget is not None:
            feasible_scores = [
                s
                for a, s in zip(all_assignments, all_scores)
                if cost_model.allocation_cost(dict(Counter(a))) <= cost_budget
            ]
            f_best = float(np.max(feasible_scores)) if feasible_scores else float("-inf")
        else:
            f_best = float(np.max(all_scores))

        for round_num in range(self.config.num_rounds):
            logger.debug(
                "Round %d/%d: SA (%d restarts x %d steps)",
                round_num + 1,
                self.config.num_rounds,
                self.config.sa.num_restarts,
                self.config.sa.steps_per_restart,
            )

            # Step 1: SA against surrogate -> top candidates
            async def surrogate_score(assignment: tuple[str, ...]) -> float:
                return surrogate.predict_mean_one(assignment)

            sa_result = await run_multi_start_sa(
                score_fn=surrogate_score,
                allocation_cost=cost_model.allocation_cost,
                feasible_allocations=feasible_allocations,
                num_layers=num_layers,
                mixer_names=mixer_names,
                cost_budget=cost_budget,
                config=self.config.sa,
                rng=rng,
                cost_floor=cost_floor,
            )
            top_candidates = [c for c, _s in sa_result.candidates]

            logger.debug("SA produced %d distinct candidates", len(top_candidates))

            # Step 2: Score with EI (full predict: mu and sigma)
            mu, sigma = surrogate.predict(top_candidates)
            ei = expected_improvement(mu, sigma, f_best, xi=self.config.ei_xi)

            # Step 3: Select top evals_per_round by EI
            n_select = min(self.config.evals_per_round, len(top_candidates))
            top_indices = np.argsort(ei)[-n_select:][::-1]

            # Emit round started event
            if self.observers:
                from place_layers.search_strategy.progress import (
                    SurrogateRoundStartedEvent,
                )

                round_started_event = SurrogateRoundStartedEvent(
                    round=round_num + 1,
                    total_rounds=self.config.num_rounds,
                    num_candidates=sa_result.total_distinct_candidates,
                    evals_planned=n_select,
                )
                for observer in self.observers:
                    observer.on_surrogate_round_started(round_started_event)

            # Record surrogate predictions for correlation diagnostic
            predicted_scores = mu[top_indices].copy()

            # Step 4: Evaluate with real evaluator
            for eval_i, idx in enumerate(top_indices):
                candidate = top_candidates[idx]
                placement = LayerPlacement(assignments=candidate)
                score = await problem.evaluate(placement)
                self.eval_count += 1

                all_assignments.append(candidate)
                all_scores.append(score)

                alloc = dict(Counter(candidate))
                candidate_cost = cost_model.allocation_cost(alloc)

                is_new_best = False
                if score > f_best:
                    f_best = score
                    is_new_best = True

                    if self.observers:
                        from place_layers.search_strategy.progress import (
                            BestUpdatedEvent,
                        )

                        event = BestUpdatedEvent(
                            score=score,
                            cost=candidate_cost,
                            allocation=alloc,
                            assignments=candidate,
                        )
                        for observer in self.observers:
                            observer.on_best_updated(event)

                if self.observers:
                    for observer in self.observers:
                        observer.on_eval_completed(self.eval_count)

                    from place_layers.search_strategy.progress import (
                        SurrogateCandidateEvaluatedEvent,
                    )

                    eval_event = SurrogateCandidateEvaluatedEvent(
                        round=round_num + 1,
                        total_rounds=self.config.num_rounds,
                        eval_index=eval_i + 1,
                        evals_planned=n_select,
                        assignments=candidate,
                        score=score,
                        surrogate_mu=float(mu[idx]),
                        ei_value=float(ei[idx]),
                        cost=candidate_cost,
                        is_new_best=is_new_best,
                        f_best=f_best,
                    )
                    for observer in self.observers:
                        observer.on_surrogate_candidate_evaluated(eval_event)

            # Compute prediction correlation for this round's candidates
            actual_scores = np.array(all_scores[-n_select:])
            if len(actual_scores) >= 2:
                corr_result = spearmanr(predicted_scores, actual_scores)
                prediction_correlation = float(corr_result.statistic)
            else:
                prediction_correlation = 0.0

            # Step 5: Refit surrogate
            surrogate.fit(all_assignments, np.array(all_scores))

            rho = surrogate.cross_validate(
                all_assignments, np.array(all_scores),
                cv_folds=self.config.cv_folds,
            )
            logger.debug(
                "Round %d complete: best=%.6f, rho=%.4f, pred_corr=%.4f",
                round_num + 1,
                f_best,
                rho,
                prediction_correlation,
            )

            # Emit round completed event
            if self.observers:
                from place_layers.search_strategy.progress import (
                    SurrogateRoundCompletedEvent,
                )

                round_completed_event = SurrogateRoundCompletedEvent(
                    round=round_num + 1,
                    total_rounds=self.config.num_rounds,
                    rho=rho,
                    prediction_correlation=prediction_correlation,
                    dataset_size=len(all_assignments),
                    best_score=f_best,
                )
                for observer in self.observers:
                    observer.on_surrogate_round_completed(round_completed_event)

        return all_assignments, all_scores


# ============================================================================
# Config
# ============================================================================


class SurrogateSearchConfig(BaseCostConstrainedConfig):
    """Configuration for surrogate-assisted search.

    The surrogate (Ridge + Bootstrap ensemble) is a drafter that proposes
    candidates cheaply.  The real evaluator (via problem.evaluate) validates
    the top candidates.

    Requires a pre-built exploration dataset from the explore_placements command.
    """

    _impl_class: ClassVar[type["BaseSearchStrategy"]] = SurrogateSearch

    strategy_type: Literal["surrogate"] = "surrogate"

    # Exploration dataset (required)
    exploration_dataset_path: str = Field(
        description="Path to JSON dataset from explore_placements command.",
    )

    cv_folds: int = Field(
        description="Cross-validation folds for surrogate quality.",
        gt=1,
    )

    # Surrogate model
    ei_xi: float = Field(
        description="Exploration parameter for Expected Improvement.",
        ge=0,
    )

    # Refinement loop
    num_rounds: int = Field(
        description="Number of surrogate-guided refinement rounds.",
        gt=0,
    )
    evals_per_round: int = Field(
        description="Top candidates by EI evaluated per round.",
        gt=0,
    )

    # SA configuration for surrogate refinement
    sa: SAConfig = Field(description="SA configuration for surrogate refinement")
