"""Surrogate-assisted search strategy for layer placement optimization.

Uses a surrogate model trained on a pre-built exploration dataset to guide
evaluation toward promising placements via acquisition function scoring.

Supports both Ridge (bootstrap + EI) and Bayesian Ridge (NIG + risk buckets)
via the ``SurrogateConfig`` discriminated union.

The surrogate is a DRAFTER: it proposes candidates cheaply,
but the real evaluator (via problem.evaluate) validates them.

Requires an exploration dataset (from explore_placements command).
Refinement: surrogate drafts candidates -> acquisition selects top-k -> real evaluator validates.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, assert_never

import numpy as np
from pydantic import Field, model_validator
from scipy.stats import spearmanr

from place_layers.annealing.sa_core import SAConfig, run_multi_start_sa
from place_layers.core.types import LayerPlacement
from place_layers.cost_model.base_cost_model import BaseCostModel
from place_layers.cost_model.enumeration import enumerate_feasible_allocations
from place_layers.problem.layer_placement import CostConstrainedProblem
from place_layers.search_strategy.cost_constrained.base import (
    BaseCostConstrainedSearch,
    feasible_best,
)
from place_layers.search_strategy.cost_constrained.base_config import (
    BaseCostConstrainedConfig,
)
from place_layers.search_strategy.cost_constrained.types import (
    BeamEntry,
    CostConstrainedSearchResult,
)
from place_layers.search_strategy.cost_constrained.risk_bucket_selection import (
    RiskBucketConfig,
    select_by_risk_buckets,
)
from place_layers.surrogate.acquisition import expected_improvement
from place_layers.surrogate.config import (
    BayesianRidgeSurrogateConfig,
    RidgeSurrogateConfig,
    SurrogateConfig,
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
# Strategy
# ============================================================================


class SurrogateSearch(BaseCostConstrainedSearch):
    """Surrogate-assisted search using SA drafting + acquisition scoring.

    Searches across the full placement space simultaneously (no per-allocation
    inner loop). Requires a pre-built exploration dataset. Refinement uses
    acquisition-guided selection of SA-generated candidates.

    Uses ``match``/``case`` on the surrogate config type to select the
    appropriate acquisition function (EI for Ridge, risk-bucket selection for Bayesian Ridge).
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

        # Find best by score (filtered to feasible entries)
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
        best_score = all_scores[best_idx]
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
        if self.config.exploration_dataset_path is None:
            raise ValueError(
                "exploration_dataset_path is null — it must be set before search. "
                "The validate_landscape command derives it per-objective at runtime."
            )

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
                source_detail="sa_surrogate",
                created_at=datetime.now(timezone.utc).isoformat(),
                num_records=None,
            )
            write_meta(eval_log_path, meta)
            eval_log_file = eval_log_path.open("a")

        try:
            # Refinement via SA against surrogate
            logger.debug("Refinement: %d rounds", self.config.num_rounds)

            # Smart constructor handles both surrogate types
            surrogate = self.config.surrogate.create_surrogate(
                mixer_names=mixer_names,
                seed=self.config.sa.seed or 0,
                feature_config=exploration.feature_config,
            )
            surrogate.fit(all_assignments, np.array(all_scores))

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

                # Step 2: Score with acquisition function — match/case on surrogate type
                mu, sigma = surrogate.predict(top_candidates)
                n_select = min(self.config.evals_per_round, len(top_candidates))
                match self.config.surrogate:
                    case RidgeSurrogateConfig(ei_xi=xi):
                        acq_scores = expected_improvement(mu, sigma, f_best, xi=xi)
                        top_indices = np.argsort(acq_scores)[-n_select:][::-1]
                    case BayesianRidgeSurrogateConfig():
                        selected_indices, bucket_log = select_by_risk_buckets(
                            mu, sigma, n_select, self.config.risk_buckets,
                        )
                        top_indices = np.array(selected_indices)
                        acq_scores = mu  # for event reporting
                        for line in bucket_log:
                            logger.info(line)
                    case other:
                        assert_never(other)

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
                            acq_value=float(acq_scores[idx]),
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

                folds = self.config.surrogate.cv_folds if isinstance(self.config.surrogate, RidgeSurrogateConfig) else None
                diag = surrogate.diagnostics(
                    all_assignments, np.array(all_scores),
                    folds=folds, rng_seed=self.config.sa.seed or 0,
                )
                label, value = diag.summary()
                logger.debug(
                    "Round %d complete: best=%.6f, %s=%.4f, pred_corr=%.4f",
                    round_num + 1,
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
                        source_detail="sa_surrogate",
                        created_at=datetime.now(timezone.utc).isoformat(),
                        num_records=eval_record_count,
                    )
                    write_meta(eval_log_path, meta_final)

        return all_assignments, all_scores


# ============================================================================
# Config
# ============================================================================


class SurrogateSearchConfig(BaseCostConstrainedConfig):
    """Configuration for surrogate-assisted search.

    The surrogate model is a drafter that proposes candidates cheaply.
    The real evaluator (via problem.evaluate) validates the top candidates.

    Supports both Ridge (bootstrap + EI) and Bayesian Ridge (NIG + risk buckets)
    via the nested ``surrogate`` config (discriminated union).

    Requires a pre-built exploration dataset from the explore_placements command.
    """

    _impl_class: ClassVar[type["BaseSearchStrategy"]] = SurrogateSearch

    strategy_type: Literal["surrogate"] = "surrogate"

    # Exploration dataset (required)
    exploration_dataset_path: str | None = Field(
        description="Path to JSON dataset from explore_placements command. "
        "Set to null in experiment YAMLs when derived at runtime.",
    )

    # Refinement loop
    num_rounds: int = Field(
        description="Number of surrogate-guided refinement rounds.",
        gt=0,
    )
    evals_per_round: int = Field(
        description="Top candidates by acquisition evaluated per round.",
        gt=0,
    )

    # SA configuration for surrogate refinement
    sa: SAConfig = Field(description="SA configuration for surrogate refinement")

    # Surrogate model configuration (discriminated union)
    surrogate: SurrogateConfig = Field(
        description="Surrogate model configuration (ridge or bayesian_ridge).",
    )

    risk_buckets: list[RiskBucketConfig] | None = Field(
        description="Risk buckets for Bayesian candidate selection. "
        "Required when surrogate is bayesian_ridge, null for ridge.",
    )

    @model_validator(mode="after")
    def _validate_risk_buckets(self) -> "SurrogateSearchConfig":
        if isinstance(self.surrogate, BayesianRidgeSurrogateConfig):
            if not self.risk_buckets:
                raise ValueError(
                    "risk_buckets must be provided when surrogate is bayesian_ridge"
                )
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
        elif self.risk_buckets is not None:
            raise ValueError(
                "risk_buckets must be null when surrogate is ridge"
            )
        return self
