"""Viterbi surrogate search strategies for layer placement optimization.

Two concrete strategies:

1. **ViterbiSurrogateSearch** (``strategy_type="viterbi_surrogate"``):
   Ridge Bootstrap Viterbi — extracts MRF potentials from each bootstrap
   Ridge model's coefficients, runs Viterbi DP on each to find exact
   optimal placements, selects via Expected Improvement.

2. **ViterbiBayesianSurrogateSearch** (``strategy_type="viterbi_bayesian_surrogate"``):
   Bayesian Viterbi with risk-bucket selection — uses exact NIG posterior
   for uncertainty, generates candidate pool via mean-only DP, then selects
   candidates using configurable risk buckets that score as
   ``mu + sign * beta * sigma`` with optional mean-floor filtering.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy.stats import spearmanr

from place_layers.core.types import LayerPlacement
from place_layers.cost_model.base_cost_model import BaseCostModel
from place_layers.cost_model.enumeration import enumerate_feasible_allocations
from place_layers.mrf.solver import ExactMRFSolver
from place_layers.problem.layer_placement import CostConstrainedProblem
from place_layers.search_strategy.cost_constrained.base import BaseCostConstrainedSearch
from place_layers.search_strategy.cost_constrained.base_config import (
    BaseCostConstrainedConfig,
)
from place_layers.search_strategy.cost_constrained.types import (
    BeamEntry,
    CostConstrainedSearchResult,
)
from place_layers.surrogate.acquisition import expected_improvement
from place_layers.surrogate.config import (
    BayesianRidgeSurrogateConfig,
    RidgeSurrogateConfig,
)
from place_layers.surrogate.dataset import ExplorationDataset

if TYPE_CHECKING:
    from place_layers.cost_model.base_cost_model import Allocation
    from place_layers.search_strategy.base_search_strategy import BaseSearchStrategy
    from place_layers.search_strategy.progress import SearchObserver
    from place_layers.search_strategy.progress.estimator import SearchProgressEstimator


logger = logging.getLogger(__name__)


# ============================================================================
# Strategy base
# ============================================================================


class _BaseViterbiSurrogateSearch(BaseCostConstrainedSearch):
    """Shared search_max_performance / search_min_cost for Viterbi surrogate strategies.

    Subclasses implement ``_run_surrogate_loop()`` with the concrete surrogate type.
    """

    config: "_BaseViterbiSurrogateConfig"

    async def _search_max_performance(
        self,
        problem: CostConstrainedProblem,
        cost_model: BaseCostModel,
        num_layers: int,
        cost_budget: float,
        cost_floor: float | None = None,
    ) -> CostConstrainedSearchResult:
        """MAX_PERFORMANCE mode: maximize performance subject to cost <= budget."""
        logger.debug("Mode: MAX_PERFORMANCE (viterbi-surrogate), budget=%.4f", cost_budget)

        feasible_allocations = enumerate_feasible_allocations(
            cost_model, num_layers, cost_budget, cost_floor=cost_floor
        )
        logger.debug("Found %d feasible allocations", len(feasible_allocations))

        if not feasible_allocations:
            raise ValueError("No feasible allocations found within budget")

        all_assignments, all_scores = await self._run_surrogate_loop(
            problem=problem,
            cost_model=cost_model,
            num_layers=num_layers,
            feasible_allocations=feasible_allocations,
            cost_budget=cost_budget,
        )

        best_idx = int(np.argmax(all_scores))
        best_assignment = all_assignments[best_idx]
        best_alloc = dict(Counter(best_assignment))
        best_entry = BeamEntry(
            assignments=best_assignment,
            allocation=best_alloc,
            score=all_scores[best_idx],
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
        logger.debug("Mode: MIN_COST (viterbi-surrogate), target=%.4f", performance_target)

        feasible_allocations = enumerate_feasible_allocations(
            cost_model, num_layers, cost_budget=None, cost_floor=cost_floor
        )
        logger.debug("Found %d total allocations", len(feasible_allocations))

        if not feasible_allocations:
            raise ValueError("No allocations found")

        all_assignments, all_scores = await self._run_surrogate_loop(
            problem=problem,
            cost_model=cost_model,
            num_layers=num_layers,
            feasible_allocations=feasible_allocations,
            cost_budget=None,
        )

        best_cost = float("inf")
        fallback_score = float("-inf")

        best_assignment: tuple[str, ...] | None = None
        best_alloc: dict[str, int] | None = None
        fallback_assignment: tuple[str, ...] | None = None
        fallback_alloc: dict[str, int] | None = None

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
        # Re-evaluate to get scalar score for BeamEntry
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
    ) -> tuple[list[tuple[str, ...]], list[float]]:
        """Load exploration data and run Viterbi refinement loop."""
        raise NotImplementedError


# ============================================================================
# Ridge Viterbi strategy
# ============================================================================


class ViterbiSurrogateSearch(_BaseViterbiSurrogateSearch):
    """Bootstrap Ridge Viterbi: MRF extraction + Viterbi DP + EI selection."""

    config: "ViterbiSurrogateSearchConfig"

    async def _run_surrogate_loop(
        self,
        problem: CostConstrainedProblem,
        cost_model: BaseCostModel,
        num_layers: int,
        feasible_allocations: list["Allocation"],
        cost_budget: float | None,
    ) -> tuple[list[tuple[str, ...]], list[float]]:
        """Bootstrap Viterbi loop."""
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

        # Initial surrogate fit
        surrogate = self.config.surrogate.create_surrogate(
            mixer_names=mixer_names,
            seed=self.config.seed,
            feature_config=exploration.feature_config,
        )
        t0 = time.perf_counter()
        surrogate.fit(all_assignments, np.array(all_scores))
        logger.info(
            "Initial surrogate fit: %.3fs (%d samples)",
            time.perf_counter() - t0,
            len(all_assignments),
        )
        f_best = float(np.max(all_scores))

        for round_num in range(self.config.num_rounds):
            logger.debug(
                "Round %d/%d: Bootstrap Viterbi (%d models x %d candidates)",
                round_num + 1,
                self.config.num_rounds,
                len(surrogate._bootstrap_models),
                self.config.candidates_per_model,
            )

            # Step 1: Extract MRF models from bootstrap ensemble
            t0 = time.perf_counter()
            mrf_models = surrogate.extract_bootstrap_mrf_models(num_layers)
            t_extract = time.perf_counter() - t0

            # Step 2: For each MRF model, run Viterbi to get globally top-M
            t0 = time.perf_counter()
            candidate_set: set[tuple[str, ...]] = set()
            for mrf_model in mrf_models:
                solver = ExactMRFSolver(mrf_model)
                top_n = solver.solve_top_n(
                    feasible_allocations, n=self.config.candidates_per_model
                )
                for solution in top_n:
                    candidate_set.add(solution.placement.assignments)
            t_viterbi = time.perf_counter() - t0

            candidate_list = list(candidate_set)
            logger.info(
                "Round %d/%d: MRF extract %.3fs, Viterbi DP %.3fs "
                "(%d models × %d allocs → %d candidates)",
                round_num + 1,
                self.config.num_rounds,
                t_extract,
                t_viterbi,
                len(mrf_models),
                len(feasible_allocations),
                len(candidate_list),
            )

            if not candidate_list:
                logger.warning("No candidates from Viterbi, skipping round")
                continue

            # Step 3: Score with EI (full predict: mu and sigma)
            mu, sigma = surrogate.predict(candidate_list)
            ei = expected_improvement(
                mu, sigma, f_best, xi=self.config.surrogate.ei_xi
            )

            # Step 4: Select top evals_per_round by EI
            n_select = min(self.config.evals_per_round, len(candidate_list))
            top_indices = np.argsort(ei)[-n_select:][::-1]

            # Emit round started event
            if self.observers:
                from place_layers.search_strategy.progress import (
                    SurrogateRoundStartedEvent,
                )

                round_started_event = SurrogateRoundStartedEvent(
                    round=round_num + 1,
                    total_rounds=self.config.num_rounds,
                    num_candidates=len(candidate_list),
                    evals_planned=n_select,
                )
                for observer in self.observers:
                    observer.on_surrogate_round_started(round_started_event)

            # Record surrogate predictions for correlation diagnostic
            predicted_scores = mu[top_indices].copy()

            # Step 5: Evaluate with real evaluator
            for eval_i, idx in enumerate(top_indices):
                candidate = candidate_list[idx]
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
                        acq_value=float(ei[idx]),
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

            # Step 6: Refit surrogate
            t0 = time.perf_counter()
            surrogate.fit(all_assignments, np.array(all_scores))
            t_refit = time.perf_counter() - t0

            t0 = time.perf_counter()
            rho = surrogate.cross_validate(
                all_assignments, np.array(all_scores),
                cv_folds=self.config.cv_folds,
            )
            t_cv = time.perf_counter() - t0
            logger.info(
                "Round %d complete: refit %.3fs, CV %.3fs, "
                "best=%.6f, rho=%.4f, pred_corr=%.4f",
                round_num + 1,
                t_refit,
                t_cv,
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
# Bayesian Viterbi strategy
# ============================================================================


class ViterbiBayesianSurrogateSearch(_BaseViterbiSurrogateSearch):
    """Bayesian Viterbi: mean-only DP pool + risk-bucket candidate selection."""

    config: "ViterbiBayesianSurrogateSearchConfig"

    async def _run_surrogate_loop(
        self,
        problem: CostConstrainedProblem,
        cost_model: BaseCostModel,
        num_layers: int,
        feasible_allocations: list["Allocation"],
        cost_budget: float | None,
    ) -> tuple[list[tuple[str, ...]], list[float]]:
        """Bayesian Viterbi loop with risk-bucket selection."""
        mixer_names = cost_model.phase_order

        # Pool sizing guard
        if self.config.candidates_per_model < self.config.evals_per_round:
            raise ValueError(
                f"candidates_per_model ({self.config.candidates_per_model}) must be "
                f">= evals_per_round ({self.config.evals_per_round})"
            )

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

        # Initial surrogate fit
        surrogate = self.config.surrogate.create_surrogate(
            mixer_names=mixer_names,
            seed=self.config.seed,
            feature_config=exploration.feature_config,
        )
        t0 = time.perf_counter()
        surrogate.fit(all_assignments, np.array(all_scores))
        logger.info(
            "Initial Bayesian surrogate fit: %.3fs (%d samples)",
            time.perf_counter() - t0,
            len(all_assignments),
        )
        f_best = float(np.max(all_scores))

        for round_num in range(self.config.num_rounds):
            logger.debug(
                "Round %d/%d: Bayesian Viterbi (%d risk buckets, %d candidates)",
                round_num + 1,
                self.config.num_rounds,
                len(self.config.risk_buckets),
                self.config.candidates_per_model,
            )

            # Step 1: Pool generation — single DP call on posterior mean MRF
            t0 = time.perf_counter()
            mrf = surrogate.extract_mrf_model(num_layers)
            t_extract = time.perf_counter() - t0

            t0 = time.perf_counter()
            solver = ExactMRFSolver(mrf)
            solutions = solver.solve_top_n(
                feasible_allocations, n=self.config.candidates_per_model
            )
            candidate_set: set[tuple[str, ...]] = set()
            for sol in solutions:
                candidate_set.add(sol.placement.assignments)
            pool = list(candidate_set)
            t_viterbi = time.perf_counter() - t0

            logger.info(
                "Round %d/%d: MRF extract %.3fs, DP %.3fs "
                "(%d allocs → %d unique candidates in pool)",
                round_num + 1,
                self.config.num_rounds,
                t_extract,
                t_viterbi,
                len(feasible_allocations),
                len(pool),
            )

            if not pool:
                logger.warning("No candidates from DP, skipping round")
                continue

            # Step 2: Predict mu/sigma for pool
            mu, sigma = surrogate.predict(pool)

            # Step 3: Select via risk_buckets
            n_select = min(self.config.evals_per_round, len(pool))
            selected: list[int] = []
            selected_set: set[int] = set()

            # Largest-remainder apportionment across buckets
            quotas = [b.quota for b in self.config.risk_buckets]
            raw = [q * n_select for q in quotas]
            base = [int(np.floor(r)) for r in raw]
            target_total = min(n_select, int(np.floor(sum(raw) + 1e-9)))
            left = max(0, target_total - sum(base))
            remainders = [(raw[i] - base[i], i) for i in range(len(base))]
            remainders.sort(reverse=True)
            for _, i in remainders[:left]:
                base[i] += 1

            bucket_selection_log: list[str] = []

            for bucket_idx, bucket in enumerate(self.config.risk_buckets):
                k_bucket = base[bucket_idx]
                if k_bucket == 0:
                    continue

                # Split k_bucket across betas with exact accounting
                kb = k_bucket // len(bucket.betas)
                kr = k_bucket % len(bucket.betas)

                # mean_floor_quantile: eligible candidates have mu >= quantile(mu, q)
                floor_val = float(np.quantile(mu, bucket.mean_floor_quantile))

                remaining_k_bucket = k_bucket

                for j, beta in enumerate(bucket.betas):
                    k = kb + (1 if j < kr else 0)
                    if k == 0:
                        continue
                    eligible = [i for i in range(len(pool))
                                if i not in selected_set and mu[i] >= floor_val]
                    if not eligible:
                        continue
                    eligible_arr = np.array(eligible)
                    scores = mu[eligible_arr] + bucket.sign * beta * sigma[eligible_arr]
                    order = np.argsort(scores, kind="mergesort")[::-1][:k]
                    added = 0
                    for idx in order:
                        if remaining_k_bucket <= 0 or len(selected) >= n_select:
                            break
                        i = int(eligible_arr[idx])
                        selected.append(i)
                        selected_set.add(i)
                        remaining_k_bucket -= 1
                        added += 1
                    bucket_selection_log.append(
                        f"  {bucket.name}(sign={bucket.sign}, β={beta}): "
                        f"selected {added}, "
                        f"mean(μ)={float(np.mean(mu[eligible_arr])):.4f}, "
                        f"mean(σ)={float(np.mean(sigma[eligible_arr])):.4f}"
                    )

                # Within-bucket backfill: if some beta sub-buckets had no eligible
                # candidates, fill remaining_k_bucket using beta=0 (pure mu)
                if remaining_k_bucket > 0:
                    eligible = [i for i in range(len(pool))
                                if i not in selected_set and mu[i] >= floor_val]
                    if eligible:
                        eligible_arr = np.array(eligible)
                        order = np.argsort(mu[eligible_arr], kind="mergesort")[::-1]
                        backfilled = 0
                        for idx in order:
                            if remaining_k_bucket <= 0 or len(selected) >= n_select:
                                break
                            i = int(eligible_arr[idx])
                            selected.append(i)
                            selected_set.add(i)
                            remaining_k_bucket -= 1
                            backfilled += 1
                        if backfilled > 0:
                            bucket_selection_log.append(
                                f"  {bucket.name}(backfill β=0): {backfilled}"
                            )

            # Global backfill by mu desc
            if len(selected) < n_select:
                remaining = [i for i in range(len(pool)) if i not in selected_set]
                remaining_arr = np.array(remaining) if remaining else np.array([], dtype=int)
                if len(remaining_arr) > 0:
                    order = np.argsort(mu[remaining_arr], kind="mergesort")[::-1]
                    global_backfilled = 0
                    for idx in order:
                        if len(selected) >= n_select:
                            break
                        selected.append(int(remaining_arr[idx]))
                        global_backfilled += 1
                    if global_backfilled > 0:
                        bucket_selection_log.append(
                            f"  global backfill (μ desc): {global_backfilled}"
                        )

            # Step 4: Log pool/selection stats
            logger.info(
                "Round %d/%d: pool=%d, n_select=%d, σ range=[%.4f, %.4f], "
                "σ mean=%.4f\n%s",
                round_num + 1,
                self.config.num_rounds,
                len(pool),
                n_select,
                float(np.min(sigma)),
                float(np.max(sigma)),
                float(np.mean(sigma)),
                "\n".join(bucket_selection_log) if bucket_selection_log else "  (no selections)",
            )

            candidate_list = [pool[i] for i in selected]

            # Emit round started event
            if self.observers:
                from place_layers.search_strategy.progress import (
                    SurrogateRoundStartedEvent,
                )

                round_started_event = SurrogateRoundStartedEvent(
                    round=round_num + 1,
                    total_rounds=self.config.num_rounds,
                    num_candidates=len(pool),
                    evals_planned=len(candidate_list),
                )
                for observer in self.observers:
                    observer.on_surrogate_round_started(round_started_event)

            # Record surrogate predictions for correlation diagnostic
            predicted_scores = mu[np.array(selected)]

            # Step 5: Evaluate with real evaluator
            for eval_i, candidate in enumerate(candidate_list):
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

                    mu_val = float(mu[selected[eval_i]])
                    eval_event = SurrogateCandidateEvaluatedEvent(
                        round=round_num + 1,
                        total_rounds=self.config.num_rounds,
                        eval_index=eval_i + 1,
                        evals_planned=len(candidate_list),
                        assignments=candidate,
                        score=score,
                        surrogate_mu=mu_val,
                        acq_value=0.0,  # N/A for risk-bucket mode
                        cost=candidate_cost,
                        is_new_best=is_new_best,
                        f_best=f_best,
                    )
                    for observer in self.observers:
                        observer.on_surrogate_candidate_evaluated(eval_event)

            # Compute prediction correlation for this round's candidates
            actual_scores_arr = np.array(all_scores[-len(candidate_list):])
            if len(actual_scores_arr) >= 2:
                corr_result = spearmanr(predicted_scores[:len(actual_scores_arr)], actual_scores_arr)
                prediction_correlation = float(corr_result.statistic)
            else:
                prediction_correlation = 0.0

            # Step 6: Refit surrogate
            t0 = time.perf_counter()
            surrogate.fit(all_assignments, np.array(all_scores))
            t_refit = time.perf_counter() - t0

            t0 = time.perf_counter()
            rho = surrogate.cross_validate(
                all_assignments, np.array(all_scores),
                cv_folds=self.config.cv_folds,
            )
            t_cv = time.perf_counter() - t0
            logger.info(
                "Round %d complete: refit %.3fs, CV %.3fs, "
                "best=%.6f, rho=%.4f, pred_corr=%.4f",
                round_num + 1,
                t_refit,
                t_cv,
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
# Configs
# ============================================================================


class _BaseViterbiSurrogateConfig(BaseCostConstrainedConfig):
    """Shared fields for both Viterbi surrogate variants."""

    exploration_dataset_path: str = Field(
        description="Path to JSON dataset from explore_placements command.",
    )

    cv_folds: int = Field(
        description="Cross-validation folds for surrogate quality.",
        gt=1,
    )

    num_rounds: int = Field(
        description="Number of surrogate-guided refinement rounds.",
        gt=0,
    )

    evals_per_round: int = Field(
        description="Top candidates evaluated per round.",
        gt=0,
    )

    seed: int = Field(
        description="Random seed for surrogate fitting.",
    )

    candidates_per_model: int = Field(
        description="Number of top placements to harvest per model (ridge) or per beta (bayesian).",
        gt=0,
    )


class ViterbiSurrogateSearchConfig(_BaseViterbiSurrogateConfig):
    """Configuration for Ridge Bootstrap Viterbi search.

    Extracts MRF potentials from bootstrap Ridge models, runs Viterbi DP
    on each to find exact optimal placements, selects via EI.

    Requires a pre-built exploration dataset from the explore_placements command.
    """

    _impl_class: ClassVar[type["BaseSearchStrategy"]] = ViterbiSurrogateSearch

    strategy_type: Literal["viterbi_surrogate"] = "viterbi_surrogate"

    surrogate: RidgeSurrogateConfig = Field(
        description="Ridge surrogate model configuration.",
    )


class RiskBucketConfig(BaseModel, frozen=True):
    """Configuration for a single risk bucket used in candidate selection.

    Each bucket scores candidates as ``mu + sign * beta * sigma`` and selects
    top-scoring ones.  The ``mean_floor_quantile`` optionally filters out
    candidates whose predicted mean is below the given quantile of the pool.
    """

    name: str = Field(description="Human-readable bucket name (e.g. 'safe', 'upside').")
    sign: Literal[-1, 1] = Field(
        description="Score sign: -1 for conservative (penalise uncertainty), +1 for optimistic.",
    )
    betas: list[float] = Field(
        description="Non-negative beta multipliers for sigma. Non-empty.",
    )
    quota: float = Field(
        description="Fraction of the per-round evaluation budget allocated to this bucket.",
        ge=0.0,
    )
    mean_floor_quantile: float = Field(
        description="Quantile of the pool mean predictions used as eligibility floor. "
                    "0.0 means no floor (all candidates eligible).",
        ge=0.0,
        le=1.0,
    )


class ViterbiBayesianSurrogateSearchConfig(_BaseViterbiSurrogateConfig):
    """Configuration for Bayesian Viterbi with risk-bucket candidate selection.

    Uses exact NIG posterior for uncertainty.  Pool generation is mean-only
    (single DP call on posterior mean MRF).  Candidate selection applies
    configurable risk buckets that score candidates as
    ``mu + sign * beta * sigma`` with optional mean-floor filtering.

    Requires a pre-built exploration dataset from the explore_placements command.
    """

    _impl_class: ClassVar[type["BaseSearchStrategy"]] = ViterbiBayesianSurrogateSearch

    strategy_type: Literal["viterbi_bayesian_surrogate"] = "viterbi_bayesian_surrogate"

    surrogate: BayesianRidgeSurrogateConfig = Field(
        description="Bayesian Ridge surrogate model configuration.",
    )

    risk_buckets: list[RiskBucketConfig] = Field(
        description="Risk buckets for post-hoc candidate selection from the DP pool.",
    )

    @model_validator(mode="after")
    def _validate_risk_buckets(self) -> "ViterbiBayesianSurrogateSearchConfig":
        if not self.risk_buckets:
            raise ValueError("risk_buckets must contain at least one bucket")
        for bucket in self.risk_buckets:
            if not bucket.betas:
                raise ValueError(
                    f"Bucket '{bucket.name}': betas must be non-empty"
                )
            if any(b < 0 for b in bucket.betas):
                raise ValueError(
                    f"Bucket '{bucket.name}': all betas must be >= 0"
                )
        total_quota = sum(b.quota for b in self.risk_buckets)
        if total_quota > 1.0 + 1e-9:
            raise ValueError(
                f"risk_buckets quota sum ({total_quota:.4f}) must be <= 1.0"
            )
        return self
