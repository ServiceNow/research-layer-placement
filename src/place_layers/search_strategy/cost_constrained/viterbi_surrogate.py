"""Viterbi surrogate search strategy for layer placement optimization.

Supports two surrogate types:

1. **ridge** (default): Bootstrap Viterbi — extracts MRF potentials from each
   bootstrap Ridge model's coefficients, runs Viterbi DP on each to find exact
   optimal placements, selects via Expected Improvement.

2. **bayesian_ridge**: Bayesian Viterbi with Pareto LCB — uses exact NIG
   posterior for uncertainty, runs Pareto DP with (mean, variance) frontier
   tracking across a sweep of β values, selects via full-posterior LCB.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.stats import spearmanr

from place_layers.core.types import LayerPlacement
from place_layers.cost_model.base_cost_model import BaseCostModel
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


class ViterbiSurrogateSearch(BaseCostConstrainedSearch):
    """Viterbi surrogate search with Ridge (bootstrap) or Bayesian Ridge (Pareto LCB).

    For surrogate_type="ridge": extracts MRF potentials from each bootstrap
    Ridge model's coefficients, runs Viterbi DP to find globally top-N
    placements per model, pools and deduplicates, selects via EI.

    For surrogate_type="bayesian_ridge": uses exact NIG posterior, runs Pareto
    DP with (mean, variance) frontier tracking across β values, selects via
    full-posterior LCB ranking.
    """

    config: "ViterbiSurrogateSearchConfig"

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
        if self.config.surrogate_type == "bayesian_ridge":
            return await self._run_bayesian_loop(
                problem, cost_model, num_layers, feasible_allocations, cost_budget
            )
        return await self._run_ridge_loop(
            problem, cost_model, num_layers, feasible_allocations, cost_budget
        )

    async def _run_ridge_loop(
        self,
        problem: CostConstrainedProblem,
        cost_model: BaseCostModel,
        num_layers: int,
        feasible_allocations: list["Allocation"],
        cost_budget: float | None,
    ) -> tuple[list[tuple[str, ...]], list[float]]:
        """Original bootstrap Viterbi loop (surrogate_type='ridge')."""
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
        surrogate = RidgeSurrogate(
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
            ei = expected_improvement(mu, sigma, f_best, xi=self.config.ei_xi)

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

    async def _run_bayesian_loop(
        self,
        problem: CostConstrainedProblem,
        cost_model: BaseCostModel,
        num_layers: int,
        feasible_allocations: list["Allocation"],
        cost_budget: float | None,
    ) -> tuple[list[tuple[str, ...]], list[float]]:
        """Bayesian Viterbi loop with Pareto LCB (surrogate_type='bayesian_ridge')."""
        from place_layers.mrf.pareto_lcb_solver import ParetoLCBSolver
        from place_layers.surrogate.bayesian_ridge import BayesianRidgeSurrogate

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
        surrogate = BayesianRidgeSurrogate(
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

        lcb_betas = self.config.lcb_betas
        lcb_quotas = self.config.lcb_quotas

        for round_num in range(self.config.num_rounds):
            logger.debug(
                "Round %d/%d: Bayesian Viterbi (β=%s, %d candidates/β)",
                round_num + 1,
                self.config.num_rounds,
                lcb_betas,
                self.config.candidates_per_model,
            )

            # Step 1: Extract count-free MRF models for DP
            t0 = time.perf_counter()
            mean_model, var_model = surrogate.extract_mean_var_mrf_models(num_layers)
            t_extract = time.perf_counter() - t0

            # Step 2: For each β, run Pareto DP → top-K candidates
            t0 = time.perf_counter()
            solver = ParetoLCBSolver(
                mean_model, var_model, surrogate.noise_scale_sq
            )
            # Track which β generated each candidate
            candidate_set: set[tuple[str, ...]] = set()
            candidates_by_beta: dict[float, list[tuple[str, ...]]] = {
                b: [] for b in lcb_betas
            }
            for beta in lcb_betas:
                solutions = solver.solve_top_n_lcb(
                    feasible_allocations,
                    n=self.config.candidates_per_model,
                    beta=beta,
                    max_frontier_size=self.config.max_frontier_size,
                )
                for sol in solutions:
                    a = sol.placement.assignments
                    candidates_by_beta[beta].append(a)
                    candidate_set.add(a)
            t_viterbi = time.perf_counter() - t0

            logger.info(
                "Round %d/%d: MRF extract %.3fs, Pareto DP %.3fs "
                "(%d β values × %d allocs → %d candidates)",
                round_num + 1,
                self.config.num_rounds,
                t_extract,
                t_viterbi,
                len(lcb_betas),
                len(feasible_allocations),
                len(candidate_set),
            )

            if not candidate_set:
                logger.warning("No candidates from Pareto DP, skipping round")
                continue

            # Step 3: Full-posterior LCB ranking with per-β quotas
            n_select = min(self.config.evals_per_round, len(candidate_set))
            selected: list[tuple[str, ...]] = []
            selected_set: set[tuple[str, ...]] = set()

            for beta, quota_frac in zip(lcb_betas, lcb_quotas):
                bucket = candidates_by_beta[beta]
                if not bucket:
                    continue
                scores = surrogate.full_lcb(bucket, beta=beta)
                k = max(1, int(np.floor(quota_frac * n_select)))
                top_idx = np.argsort(scores)[::-1]
                added_this_beta = 0
                for idx in top_idx:
                    if len(selected) >= n_select:
                        break
                    if added_this_beta >= k:
                        break
                    a = bucket[idx]
                    if a not in selected_set:
                        selected.append(a)
                        selected_set.add(a)
                        added_this_beta += 1

            # Fill remaining slots if quotas didn't cover all
            if len(selected) < n_select:
                remaining = [a for a in candidate_set if a not in selected_set]
                if remaining:
                    scores = surrogate.full_lcb(remaining, beta=self.config.ranking_beta)
                    for idx in np.argsort(scores)[::-1]:
                        if len(selected) >= n_select:
                            break
                        selected.append(remaining[idx])

            candidate_list = selected[:n_select]

            # Emit round started event
            if self.observers:
                from place_layers.search_strategy.progress import (
                    SurrogateRoundStartedEvent,
                )

                round_started_event = SurrogateRoundStartedEvent(
                    round=round_num + 1,
                    total_rounds=self.config.num_rounds,
                    num_candidates=len(candidate_set),
                    evals_planned=len(candidate_list),
                )
                for observer in self.observers:
                    observer.on_surrogate_round_started(round_started_event)

            # Record surrogate predictions for correlation diagnostic
            predicted_scores = surrogate.predict_mean(candidate_list)

            # Step 4: Evaluate with real evaluator
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

                    mu_val = float(surrogate.predict_mean_one(candidate))
                    eval_event = SurrogateCandidateEvaluatedEvent(
                        round=round_num + 1,
                        total_rounds=self.config.num_rounds,
                        eval_index=eval_i + 1,
                        evals_planned=len(candidate_list),
                        assignments=candidate,
                        score=score,
                        surrogate_mu=mu_val,
                        acq_value=0.0,  # N/A for Bayesian mode
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

            # Step 5: Refit surrogate
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
# Config
# ============================================================================


class ViterbiSurrogateSearchConfig(BaseCostConstrainedConfig):
    """Configuration for Viterbi surrogate search.

    Supports two modes:
    - surrogate_type="ridge": Bootstrap Viterbi with EI selection (original).
    - surrogate_type="bayesian_ridge": Bayesian Viterbi with Pareto LCB.

    Requires a pre-built exploration dataset from the explore_placements command.
    """

    _impl_class: ClassVar[type["BaseSearchStrategy"]] = ViterbiSurrogateSearch

    strategy_type: Literal["viterbi_surrogate"] = "viterbi_surrogate"

    exploration_dataset_path: str = Field(
        description="Path to JSON dataset from explore_placements command.",
    )

    cv_folds: int = Field(
        description="Cross-validation folds for surrogate quality.",
        gt=1,
    )

    ei_xi: float = Field(
        default=0.01,
        description="Exploration parameter for Expected Improvement (ridge mode only).",
        ge=0,
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
        description="Number of top placements to harvest per model (ridge) or per β (bayesian_ridge).",
        gt=0,
    )

    surrogate_type: Literal["ridge", "bayesian_ridge"] = Field(
        default="ridge",
        description="Surrogate model type: 'ridge' (bootstrap) or 'bayesian_ridge' (NIG posterior).",
    )

    lcb_betas: list[float] = Field(
        default=[0.0, 0.5, 1.0, 2.0],
        description="LCB β values for Bayesian mode. Higher β → more conservative (stronger uncertainty penalty).",
    )

    lcb_quotas: list[float] = Field(
        default=[0.4, 0.3, 0.2, 0.1],
        description="Per-β evaluation budget fractions for Bayesian mode.",
    )

    ranking_beta: float = Field(
        default=0.0,
        description="β used for full-LCB ranking of fill candidates after per-β quotas. "
                    "0.0 = rank by mean only.",
        ge=0.0,
    )

    max_frontier_size: int = Field(
        default=500,
        description="Safety cap on Pareto frontier size per DP state (Bayesian mode).",
        gt=0,
    )

    @model_validator(mode="after")
    def _validate_lcb_config(self) -> "ViterbiSurrogateSearchConfig":
        if len(self.lcb_betas) != len(self.lcb_quotas):
            raise ValueError(
                f"lcb_betas ({len(self.lcb_betas)}) and lcb_quotas "
                f"({len(self.lcb_quotas)}) must have equal length"
            )
        if any(q < 0 for q in self.lcb_quotas):
            raise ValueError("All lcb_quotas must be non-negative")
        if sum(self.lcb_quotas) > 1.0 + 1e-9:
            raise ValueError(
                f"lcb_quotas sum ({sum(self.lcb_quotas):.4f}) must be <= 1.0"
            )
        return self
