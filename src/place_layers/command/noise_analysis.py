"""Noise degradation analysis command for cost-constrained search.

This command analyzes how evaluation noise degrades search quality by:
1. Computing exact optimal placement using DP solver over all feasible allocations
2. Running cost-constrained search with noisy evaluator
3. Evaluating results using TRUE scores (no noise)
4. Reporting regret metrics across noise levels
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, assert_never, cast

import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.evaluator.evaluators import EvaluatorsConfig, Evaluators
from place_layers.evaluator.mock_supernet_evaluator import MockSupernetEvaluator
from place_layers.cost_model.base_cost_model import BaseCostModel
from place_layers.mrf import ExactMRFSolver, ExactSolution
from place_layers.problem.layer_placement import (
    CostConstrainedProblemConfig,
    MaxPerformanceProblemConfig,
    MinCostProblemConfig,
)
from place_layers.search_strategy.cost_constrained import CostConstrainedConfig
from place_layers.search_strategy.cost_constrained.types import (
    CostConstrainedSearchResult,
)


logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """Result from a single cost-constrained search trial."""

    noise_sigma: float
    noise_seed_shift: int
    optimal_true_score: float
    found_true_score: float
    found_noisy_score: float
    found_cost: float
    regret: float  # optimal_true - found_true
    relative_regret: float  # regret / |optimal_true|
    edit_distance: int  # layers differing from optimal
    found_optimal: bool  # whether search found exact optimum
    allocations_explored: int


@dataclass
class SweepResults:
    """Aggregated results from a noise analysis sweep."""

    trials: list[TrialResult]
    noise_sigmas: list[float]

    def _matching_trials(self, noise_sigma: float) -> list[TrialResult]:
        return [t for t in self.trials if t.noise_sigma == noise_sigma]

    def mean_regret(self, noise_sigma: float) -> float:
        """Compute mean regret for given noise level."""
        matching = [t.regret for t in self._matching_trials(noise_sigma)]
        return float(np.mean(matching)) if matching else float("nan")

    def std_regret(self, noise_sigma: float) -> float:
        """Compute std of regret for given noise level."""
        matching = [t.regret for t in self._matching_trials(noise_sigma)]
        return float(np.std(matching)) if matching else float("nan")

    def mean_relative_regret(self, noise_sigma: float) -> float:
        """Compute mean relative regret for given noise level."""
        matching = [t.relative_regret for t in self._matching_trials(noise_sigma)]
        return float(np.mean(matching)) if matching else float("nan")

    def mean_edit_distance(self, noise_sigma: float) -> float:
        """Compute mean edit distance for given noise level."""
        matching = [t.edit_distance for t in self._matching_trials(noise_sigma)]
        return float(np.mean(matching)) if matching else float("nan")

    def std_edit_distance(self, noise_sigma: float) -> float:
        """Compute std of edit distance for given noise level."""
        matching = [t.edit_distance for t in self._matching_trials(noise_sigma)]
        return float(np.std(matching)) if matching else float("nan")


class NoiseAnalysisCommand(BaseCommand):
    """Analyze how noise degrades cost-constrained search performance.

    This command uses the MockSupernetEvaluator's MRF structure to:
    1. Compute the exact optimal placement using Viterbi DP over feasible allocations
    2. Run cost-constrained search multiple times with varying noise levels
    3. Measure regret (difference between optimal and achieved true scores)
    """

    config: "NoiseAnalysisCommandConfig"

    def _compute_exact_optimal(
        self, solver: ExactMRFSolver, cost_model: BaseCostModel
    ) -> ExactSolution | None:
        """Find the true optimal placement across all feasible allocations.

        For MaxPerformance: best true score among allocations with cost <= budget.
        For MinCost: cheapest allocation achieving performance >= target.
        """
        num_layers = solver.model.num_layers
        total_allocations = cost_model.count_allocations(num_layers)
        logger.info(
            "Exact DP: %d total allocations (%d layers, %d mixers)",
            total_allocations,
            num_layers,
            cost_model.num_mixers,
        )

        cost_floor = self.config.problem.cost_floor

        match self.config.problem:
            case MaxPerformanceProblemConfig(cost_budget=budget):
                best = None
                checked = 0
                for alloc_cost, allocation in cost_model.allocations_by_cost(
                    num_layers, "decreasing"
                ):
                    if alloc_cost > budget:
                        continue
                    if cost_floor is not None and alloc_cost < cost_floor - 1e-9:
                        break
                    checked += 1
                    solution = solver.solve_constrained(allocation)
                    if solution and (
                        best is None or solution.true_score > best.true_score
                    ):
                        best = solution
                    if checked % 500 == 0:
                        logger.info(
                            "  checked %d feasible allocations (best=%.4f)…",
                            checked,
                            best.true_score if best else float("-inf"),
                        )
                logger.info("  checked %d feasible allocations total", checked)
                return best

            case MinCostProblemConfig(performance_target=target):
                checked = 0
                for alloc_cost, allocation in cost_model.allocations_by_cost(
                    num_layers, "increasing"
                ):
                    if cost_floor is not None and alloc_cost < cost_floor - 1e-9:
                        continue
                    checked += 1
                    solution = solver.solve_constrained(allocation)
                    if solution and solution.true_score >= target:
                        logger.info(
                            "  found feasible solution after %d allocations",
                            checked,
                        )
                        return solution
                logger.info("  exhausted %d allocations, no feasible solution", checked)
                return None
            case other:
                assert_never(other)

    async def run(self) -> None:
        """Execute noise analysis."""
        output_dir = Path(HydraConfig.get().runtime.output_dir)

        async with self.config.evaluators.create_evaluators(output_dir) as evals:
            # Validate all evaluators are MockSupernetEvaluator
            for name, evaluator in evals.evaluators.items():
                if not isinstance(evaluator, MockSupernetEvaluator):
                    raise TypeError(
                        f"Evaluator '{name}' must be MockSupernetEvaluator, "
                        f"got {type(evaluator).__name__}"
                    )

            # Pick any evaluator to compute exact optimal (all share same MRF
            # landscape when they have the same seed/structure params — they
            # differ only in noise_sigma / noise_seed_shift).
            first_name = next(iter(evals.evaluators))
            first_eval = cast(
                MockSupernetEvaluator, evals.evaluators[first_name]
            )
            cost_model = self.config.problem.cost_model.create_cost_model()

            logger.info("Computing exact optimal via DP…")
            solver = first_eval.create_solver()
            optimal = self._compute_exact_optimal(solver, cost_model)
            if optimal is None:
                raise ValueError("No feasible placement found")
            logger.info(
                "Exact optimal: true_score=%.4f, placement=%s…",
                optimal.true_score,
                optimal.placement.assignments[:10],
            )
            optimal_true = optimal.true_score

            # Run search for each evaluator (each has its own noise config)
            all_trials: list[TrialResult] = []
            total = len(evals.evaluators)
            for trial_num, name in enumerate(evals.evaluators, 1):
                evaluator = cast(
                    MockSupernetEvaluator, evals.evaluators[name]
                )
                sigma = evaluator.config.noise_sigma
                seed_shift = evaluator.config.noise_seed_shift

                logger.info(
                    "[%d/%d] %s: sigma=%.2f, seed_shift=%d",
                    trial_num, total, name, sigma, seed_shift,
                )

                # Each evaluator name is an objective name (IndependentObjective)
                problem = self.config.problem.create_problem(
                    evals, objective_name=name
                )
                strategy = self.config.search_strategy.create_search_strategy()
                result = cast(
                    CostConstrainedSearchResult, await strategy.search(problem)
                )

                found_true = first_eval.true_score(result.best_parameters)
                regret = optimal_true - found_true
                edit_distance = sum(
                    a != b
                    for a, b in zip(
                        result.best_parameters.assignments,
                        optimal.placement.assignments,
                        strict=True,
                    )
                )

                all_trials.append(
                    TrialResult(
                        noise_sigma=sigma,
                        noise_seed_shift=seed_shift,
                        optimal_true_score=optimal_true,
                        found_true_score=found_true,
                        found_noisy_score=result.best_score,
                        found_cost=result.final_cost,
                        regret=regret,
                        relative_regret=(
                            abs(regret / optimal_true)
                            if optimal_true != 0
                            else 0.0
                        ),
                        edit_distance=edit_distance,
                        found_optimal=edit_distance == 0,
                        allocations_explored=result.allocations_explored,
                    )
                )

            # Derive noise_sigmas from trials
            noise_sigmas = sorted(set(t.noise_sigma for t in all_trials))

            sweep = SweepResults(trials=all_trials, noise_sigmas=noise_sigmas)
            self._save_results(sweep, output_dir, evals)
            self._create_plots(sweep, output_dir)
            self._print_summary(sweep)

        logger.info("Results saved to %s", output_dir)

    def _save_results(
        self,
        results: SweepResults,
        output_dir: Path,
        evals: Evaluators,
    ) -> None:
        """Save results to JSON."""
        data = {
            "config": {
                "num_layers": evals.num_layers,
                "mixer_names": sorted(evals.mixer_names),
                "problem_type": self.config.problem.problem_type,
                "search_strategy_type": self.config.search_strategy.strategy_type,
                "noise_sigmas": results.noise_sigmas,
                "num_trials": len(results.trials),
            },
            "trials": [
                {
                    "noise_sigma": t.noise_sigma,
                    "noise_seed_shift": t.noise_seed_shift,
                    "optimal_true_score": t.optimal_true_score,
                    "found_true_score": t.found_true_score,
                    "found_noisy_score": t.found_noisy_score,
                    "found_cost": t.found_cost,
                    "regret": t.regret,
                    "relative_regret": t.relative_regret,
                    "edit_distance": t.edit_distance,
                    "found_optimal": t.found_optimal,
                    "allocations_explored": t.allocations_explored,
                }
                for t in results.trials
            ],
        }

        results_path = output_dir / "noise_analysis_results.json"
        with open(results_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved results to %s", results_path)

    def _create_plots(self, results: SweepResults, output_dir: Path) -> None:
        """Create visualization plots."""
        sigmas = results.noise_sigmas
        x = np.arange(len(sigmas))
        sigma_labels = [f"{s:.2f}" for s in sigmas]

        # Plot 1: Bar charts — mean regret and mean edit distance
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        mean_regrets = [results.mean_regret(s) for s in sigmas]
        ax = axes[0]
        ax.bar(x, mean_regrets, color="salmon")
        ax.set_xticks(x)
        ax.set_xticklabels(sigma_labels)
        ax.set_xlabel("Noise Sigma")
        ax.set_ylabel("Mean Regret")
        ax.set_title("Mean Regret vs Noise Level")

        mean_edit_dists = [results.mean_edit_distance(s) for s in sigmas]
        ax = axes[1]
        ax.bar(x, mean_edit_dists, color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels(sigma_labels)
        ax.set_xlabel("Noise Sigma")
        ax.set_ylabel("Mean Edit Distance")
        ax.set_title("Mean Edit Distance vs Noise Level")

        plt.tight_layout()
        plt.savefig(output_dir / "noise_analysis_bars.png", dpi=150)
        logger.info("Saved bar charts to %s", output_dir / "noise_analysis_bars.png")

        # Plot 2: Line plots — regret and edit distance with error bars
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        std_regrets = [results.std_regret(s) for s in sigmas]
        ax1.errorbar(
            sigmas, mean_regrets, yerr=std_regrets,
            marker="o", capsize=4, linewidth=2, color="salmon",
        )
        ax1.set_xlabel("Noise Sigma")
        ax1.set_ylabel("Regret")
        ax1.set_title("Regret vs Noise Level")
        ax1.grid(True, alpha=0.3)

        std_edit_dists = [results.std_edit_distance(s) for s in sigmas]
        ax2.errorbar(
            sigmas, mean_edit_dists, yerr=std_edit_dists,
            marker="s", capsize=4, linewidth=2, color="steelblue",
        )
        ax2.set_xlabel("Noise Sigma")
        ax2.set_ylabel("Edit Distance")
        ax2.set_title("Edit Distance vs Noise Level")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "noise_analysis_lines.png", dpi=150)
        logger.info("Saved line plot to %s", output_dir / "noise_analysis_lines.png")
        plt.close("all")

    def _print_summary(self, results: SweepResults) -> None:
        """Print summary statistics."""
        logger.info(
            "Summary: sigma | mean_regret | std_regret | mean_edit_dist | std_edit_dist",
        )
        for sigma in results.noise_sigmas:
            logger.info(
                "  %5.2f | %11.3f | %10.3f | %14.1f | %13.1f",
                sigma,
                results.mean_regret(sigma),
                results.std_regret(sigma),
                results.mean_edit_distance(sigma),
                results.std_edit_distance(sigma),
            )


class NoiseAnalysisCommandConfig(BaseCommandConfig):
    """Configuration for noise degradation analysis.

    Each evaluator entry in the standard ``EvaluatorsConfig`` should be a
    ``MockSupernetEvaluator`` with its own ``noise_sigma`` and
    ``noise_seed_shift``.  Use ``objective_type: independent``.
    """

    _impl_class: ClassVar[type[BaseCommand]] = NoiseAnalysisCommand

    command: Literal["noise_analysis"] = "noise_analysis"

    evaluators: EvaluatorsConfig

    problem: CostConstrainedProblemConfig

    search_strategy: CostConstrainedConfig
