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
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
from pydantic import Field, model_validator  # noqa: F401 (model_validator used in config)
from scipy.stats import spearmanr

from place_layers.core.types import LayerPlacement
from place_layers.cost_model.base_cost_model import BaseCostModel
from place_layers.cost_model.enumeration import enumerate_feasible_allocations
from place_layers.mrf.solver import ExactMRFSolver
from place_layers.problem.layer_placement import CostConstrainedProblem
from place_layers.search_strategy.cost_constrained.base import (
    BaseCostConstrainedSearch,
    feasible_best,
)
from place_layers.search_strategy.cost_constrained.base_config import (
    BaseCostConstrainedConfig,
)
from place_layers.search_strategy.cost_constrained.risk_bucket_selection import (
    RiskBucketConfig,
    select_by_risk_buckets,
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
from place_layers.surrogate.evaluation_log import (
    EvaluationRecord,
    append_record,
)

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
            cost_floor=cost_floor,
        )

        fb = feasible_best(
            all_assignments, all_scores, cost_model, cost_budget, cost_floor,
        )
        assert fb is not None, (
            f"BUG: no feasible placement in [{cost_floor}, {cost_budget}] "
            f"after {len(all_assignments)} samples — "
            f"this should be impossible since the surrogate loop generates in-band candidates"
        )
        best_idx, _ = fb
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
            cost_floor=cost_floor,
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
        cost_floor: float | None = None,
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
        cost_floor: float | None = None,
    ) -> tuple[list[tuple[str, ...]], list[float]]:
        """Bootstrap Viterbi loop."""
        mixer_names = cost_model.phase_order

        if self.config.exploration_dataset_path is None:
            raise ValueError(
                "exploration_dataset_path is null — it must be set before search. "
                "The validate_landscape command derives it per-objective at runtime."
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

        # Read pre-seeded records from evaluation log (written by command)
        eval_log_path = (
            Path(self.config.evaluation_log_path)
            if self.config.evaluation_log_path
            else None
        )
        if eval_log_path is not None and eval_log_path.exists():
            from place_layers.surrogate.evaluation_log import load_records

            existing_records = load_records(eval_log_path)
            if existing_records:
                n_before = len(all_assignments)
                existing = {(a, s) for a, s in zip(all_assignments, all_scores)}
                for rec in existing_records:
                    if (rec.assignment, rec.score) not in existing:
                        all_assignments.append(rec.assignment)
                        all_scores.append(rec.score)
                        existing.add((rec.assignment, rec.score))
                logger.info(
                    "Loaded %d pre-seeded records from eval log (%d new after dedup)",
                    len(existing_records),
                    len(all_assignments) - n_before,
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

        # Open evaluation log for appending strategy's own evaluations
        from place_layers.surrogate.evaluation_log import (
            EvaluationLogMeta,
            write_meta,
        )

        eval_log_file = None
        eval_record_count = 0

        if eval_log_path is not None:
            eval_log_path.parent.mkdir(parents=True, exist_ok=True)
            meta = EvaluationLogMeta(
                version=1,
                objective_name=exploration.objective_name,
                num_layers=num_layers,
                mixer_names=list(mixer_names),
                source_command="surrogate_search",
                source_detail="ridge_viterbi",
                created_at=datetime.now(timezone.utc).isoformat(),
                num_records=None,
            )
            write_meta(eval_log_path, meta)
            eval_log_file = eval_log_path.open("a")

        try:
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
            if cost_budget is not None:
                fb = feasible_best(
                    all_assignments, all_scores, cost_model, cost_budget,
                    cost_floor,
                )
                f_best = fb[1] if fb is not None else float("-inf")
            else:
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

                    # Write to evaluation log
                    if eval_log_file is not None:
                        record = EvaluationRecord(
                            assignment=candidate,
                            score=score,
                            allocation=alloc,
                        )
                        append_record(eval_log_file, record)
                        eval_record_count += 1

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
                diag = surrogate.diagnostics(
                    all_assignments, np.array(all_scores),
                    folds=self.config.surrogate.cv_folds, rng_seed=self.config.seed,
                )
                label, value = diag.summary()
                t_diag = time.perf_counter() - t0
                logger.info(
                    "Round %d complete: refit %.3fs, diag %.3fs, "
                    "best=%.6f, %s=%.4f, pred_corr=%.4f",
                    round_num + 1,
                    t_refit,
                    t_diag,
                    f_best,
                    label,
                    value,
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
                        surrogate_metric_label=label,
                        surrogate_metric_value=value,
                        diagnostics=diag,
                        prediction_correlation=prediction_correlation,
                        dataset_size=len(all_assignments),
                        best_score=f_best,
                    )
                    for observer in self.observers:
                        observer.on_surrogate_round_completed(round_completed_event)

        finally:
            # Close evaluation log and update metadata with record count
            if eval_log_file is not None:
                eval_log_file.close()
                if eval_log_path is not None and eval_record_count > 0:
                    meta_final = EvaluationLogMeta(
                        version=1,
                        objective_name=exploration.objective_name,
                        num_layers=num_layers,
                        mixer_names=list(mixer_names),
                        source_command="surrogate_search",
                        source_detail="ridge_viterbi",
                        created_at=datetime.now(timezone.utc).isoformat(),
                        num_records=eval_record_count,
                    )
                    write_meta(eval_log_path, meta_final)

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
        cost_floor: float | None = None,
    ) -> tuple[list[tuple[str, ...]], list[float]]:
        """Bayesian Viterbi loop with risk-bucket selection."""
        mixer_names = cost_model.phase_order

        if self.config.exploration_dataset_path is None:
            raise ValueError(
                "exploration_dataset_path is null — it must be set before search. "
                "The validate_landscape command derives it per-objective at runtime."
            )

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

        # Read pre-seeded records from evaluation log (written by command)
        eval_log_path = (
            Path(self.config.evaluation_log_path)
            if self.config.evaluation_log_path
            else None
        )
        if eval_log_path is not None and eval_log_path.exists():
            from place_layers.surrogate.evaluation_log import load_records as _load_records

            existing_records = _load_records(eval_log_path)
            if existing_records:
                n_before = len(all_assignments)
                existing = {(a, s) for a, s in zip(all_assignments, all_scores)}
                for rec in existing_records:
                    if (rec.assignment, rec.score) not in existing:
                        all_assignments.append(rec.assignment)
                        all_scores.append(rec.score)
                        existing.add((rec.assignment, rec.score))
                logger.info(
                    "Loaded %d pre-seeded records from eval log (%d new after dedup)",
                    len(existing_records),
                    len(all_assignments) - n_before,
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

        # Open evaluation log for appending strategy's own evaluations
        from place_layers.surrogate.evaluation_log import (
            EvaluationLogMeta,
            write_meta,
        )

        eval_log_file = None
        eval_record_count = 0

        if eval_log_path is not None:
            eval_log_path.parent.mkdir(parents=True, exist_ok=True)
            meta = EvaluationLogMeta(
                version=1,
                objective_name=exploration.objective_name,
                num_layers=num_layers,
                mixer_names=list(mixer_names),
                source_command="surrogate_search",
                source_detail="bayesian_viterbi",
                created_at=datetime.now(timezone.utc).isoformat(),
                num_records=None,
            )
            write_meta(eval_log_path, meta)
            eval_log_file = eval_log_path.open("a")

        try:
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
            if cost_budget is not None:
                fb = feasible_best(
                    all_assignments, all_scores, cost_model, cost_budget,
                    cost_floor,
                )
                f_best = fb[1] if fb is not None else float("-inf")
            else:
                f_best = float(np.max(all_scores))

            for round_num in range(self.config.num_rounds):
                logger.debug(
                    "Round %d/%d: Bayesian Viterbi (%d risk buckets, %d candidates)",
                    round_num + 1,
                    self.config.num_rounds,
                    len(self.config.risk_buckets),
                    self.config.candidates_per_model,
                )

                # Step 1: Pool generation — per-allocation DP (allocation-diverse)
                t0 = time.perf_counter()
                mrf = surrogate.extract_mrf_model(num_layers)
                t_extract = time.perf_counter() - t0

                t0 = time.perf_counter()
                solver = ExactMRFSolver(mrf)
                k_per_alloc = int(math.ceil(
                    self.config.candidates_per_model / len(feasible_allocations)
                ))
                per_alloc = solver.solve_top_n_per_allocation(
                    feasible_allocations, n_per_allocation=k_per_alloc
                )
                candidate_set: set[tuple[str, ...]] = set()
                for sols in per_alloc.values():
                    candidate_set.update(sol.placement.assignments for sol in sols)

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
                selected, bucket_selection_log = select_by_risk_buckets(
                    mu, sigma, n_select, self.config.risk_buckets,
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

                    # Write to evaluation log
                    if eval_log_file is not None:
                        record = EvaluationRecord(
                            assignment=candidate,
                            score=score,
                            allocation=alloc,
                        )
                        append_record(eval_log_file, record)
                        eval_record_count += 1

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
                diag = surrogate.diagnostics(
                    all_assignments, np.array(all_scores),
                    folds=None, rng_seed=self.config.seed,
                )
                label, value = diag.summary()
                t_diag = time.perf_counter() - t0
                logger.info(
                    "Round %d complete: refit %.3fs, diag %.3fs, "
                    "best=%.6f, %s=%.4f, pred_corr=%.4f",
                    round_num + 1,
                    t_refit,
                    t_diag,
                    f_best,
                    label,
                    value,
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
                        surrogate_metric_label=label,
                        surrogate_metric_value=value,
                        diagnostics=diag,
                        prediction_correlation=prediction_correlation,
                        dataset_size=len(all_assignments),
                        best_score=f_best,
                    )
                    for observer in self.observers:
                        observer.on_surrogate_round_completed(round_completed_event)

        finally:
            # Close evaluation log and update metadata with record count
            if eval_log_file is not None:
                eval_log_file.close()
                if eval_log_path is not None and eval_record_count > 0:
                    meta_final = EvaluationLogMeta(
                        version=1,
                        objective_name=exploration.objective_name,
                        num_layers=num_layers,
                        mixer_names=list(mixer_names),
                        source_command="surrogate_search",
                        source_detail="bayesian_viterbi",
                        created_at=datetime.now(timezone.utc).isoformat(),
                        num_records=eval_record_count,
                    )
                    write_meta(eval_log_path, meta_final)

        return all_assignments, all_scores


# ============================================================================
# Configs
# ============================================================================


class _BaseViterbiSurrogateConfig(BaseCostConstrainedConfig):
    """Shared fields for both Viterbi surrogate variants."""

    exploration_dataset_path: str | None = Field(
        description="Path to JSON dataset from explore_placements command. "
        "Set to null in experiment YAMLs when derived at runtime.",
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
