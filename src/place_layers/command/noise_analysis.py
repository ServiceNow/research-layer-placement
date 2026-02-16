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
from typing import ClassVar, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig
from pydantic import Field

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.core.types import LayerPlacement
from place_layers.evaluator.mock_supernet_evaluator import (
    MockSupernetEvaluator,
    MockSupernetEvaluatorConfig,
)
from place_layers.cost_model.base_cost_model import BaseCostModel
from place_layers.mrf import ChainMRFModel, ExactMRFSolver, ExactSolution
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
        self, mrf_model: ChainMRFModel, cost_model: BaseCostModel
    ) -> ExactSolution | None:
        """Find the true optimal placement across all feasible allocations.

        For MaxPerformance: best true score among allocations with cost <= budget.
        For MinCost: cheapest allocation achieving performance >= target.
        """
        solver = ExactMRFSolver(mrf_model)
        num_layers = mrf_model.num_layers
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
            case _:
                raise ValueError(
                    f"Unsupported problem type: {self.config.problem.problem_type}"
                )

    async def run(self) -> None:
        """Execute noise analysis."""
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)

        total_trials = len(self.config.noise_sigmas) * self.config.num_seeds
        logger.info(
            "Noise analysis: %d layers, %s mixers, %d sigmas x %d seeds = %d trials",
            self.config.evaluator.num_layers,
            self.config.evaluator.mixer_names,
            len(self.config.noise_sigmas),
            self.config.num_seeds,
            total_trials,
        )

        # Build base evaluator and extract MRF model
        logger.info("Building base evaluator (seed=%d)…", self.config.evaluator.seed)
        evaluator_config = self.config.evaluator
        assert isinstance(evaluator_config, MockSupernetEvaluatorConfig)
        base_evaluator = evaluator_config.create_evaluator()
        assert isinstance(base_evaluator, MockSupernetEvaluator)
        mrf_model = base_evaluator.mrf_model

        # Create cost model for ground truth computation
        cost_model = self.config.problem.cost_model.create_cost_model()

        # Compute exact cost-constrained optimal via DP
        logger.info("Computing exact optimal via DP…")
        optimal = self._compute_exact_optimal(mrf_model, cost_model)
        if optimal is None:
            raise ValueError(
                "No feasible placement found for the given problem configuration"
            )
        logger.info(
            "Exact optimal: true_score=%.4f, placement=%s…",
            optimal.true_score,
            optimal.placement.assignments[:10],
        )

        # Run sweep
        logger.info("Starting sweep over %d noise sigmas…", len(self.config.noise_sigmas))
        results = await self._run_sweep(
            base_evaluator=base_evaluator,
            optimal_true_score=optimal.true_score,
            optimal_placement=optimal.placement,
        )

        # Save results and create plots
        self._save_results(results, output_dir)
        self._create_plots(results, output_dir)
        self._print_summary(results)

        logger.info("Results saved to %s", output_dir)

    async def _run_sweep(
        self,
        base_evaluator: MockSupernetEvaluator,
        optimal_true_score: float,
        optimal_placement: LayerPlacement,
    ) -> SweepResults:
        """Run sweep over noise levels."""
        trials: list[TrialResult] = []

        total_trials = len(self.config.noise_sigmas) * self.config.num_seeds
        trial_num = 0

        for noise_sigma in self.config.noise_sigmas:
            logger.info("Sigma %.2f (%d seeds)", noise_sigma, self.config.num_seeds)
            for seed_offset in range(self.config.num_seeds):
                trial_num += 1
                noisy_evaluator = MockSupernetEvaluator.with_noise(
                    base_evaluator, noise_sigma, noise_seed_shift=seed_offset
                )

                result = await self._run_trial(
                    noisy_evaluator,
                    optimal_true_score=optimal_true_score,
                    optimal_placement=optimal_placement,
                )
                trials.append(result)

                logger.info(
                    "  [%d/%d] regret=%.4f (%.2f%%) edit_dist=%d",
                    trial_num,
                    total_trials,
                    result.regret,
                    result.relative_regret * 100,
                    result.edit_distance,
                )

        return SweepResults(
            trials=trials,
            noise_sigmas=self.config.noise_sigmas,
        )

    async def _run_trial(
        self,
        evaluator: MockSupernetEvaluator,
        optimal_true_score: float,
        optimal_placement: LayerPlacement,
    ) -> TrialResult:
        """Run a single cost-constrained search trial."""
        problem = self.config.problem.create_problem(evaluator)
        strategy = self.config.search_strategy.create_search_strategy()

        result = cast(CostConstrainedSearchResult, await strategy.search(problem))

        # Compute true score of found placement
        found_true_score = await evaluator.evaluate(
            result.best_parameters, noise_sigma=0.0
        )

        # Compute regret
        regret = optimal_true_score - found_true_score
        relative_regret = (
            abs(regret / optimal_true_score) if optimal_true_score != 0 else 0.0
        )

        edit_distance = sum(
            a != b
            for a, b in zip(
                result.best_parameters.assignments,
                optimal_placement.assignments,
                strict=True,
            )
        )

        return TrialResult(
            noise_sigma=evaluator.config.noise_sigma,
            noise_seed_shift=evaluator.config.noise_seed_shift,
            optimal_true_score=optimal_true_score,
            found_true_score=found_true_score,
            found_noisy_score=result.best_score,
            found_cost=result.final_cost,
            regret=regret,
            relative_regret=relative_regret,
            edit_distance=edit_distance,
            found_optimal=edit_distance == 0,
            allocations_explored=result.allocations_explored,
        )

    def _save_results(self, results: SweepResults, output_dir: Path) -> None:
        """Save results to JSON."""
        data = {
            "config": {
                "num_layers": self.config.evaluator.num_layers,
                "mixer_names": self.config.evaluator.mixer_names,
                "seed": self.config.evaluator.seed,
                "problem_type": self.config.problem.problem_type,
                "search_strategy_type": self.config.search_strategy.strategy_type,
                "noise_sigmas": self.config.noise_sigmas,
                "num_seeds": self.config.num_seeds,
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
            "Summary: sigma | mean_regret | std_regret | mean_edit_dist | std_edit_dist"
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
    """Configuration for noise degradation analysis."""

    _impl_class: ClassVar[type[BaseCommand]] = NoiseAnalysisCommand

    command: Literal["noise_analysis"] = "noise_analysis"

    evaluator: MockSupernetEvaluatorConfig = Field(
        description="Mock supernet evaluator configuration (defines the MRF landscape)",
    )

    problem: CostConstrainedProblemConfig = Field(
        description="Cost-constrained problem configuration (cost model + constraints)",
    )

    search_strategy: CostConstrainedConfig = Field(
        description="Cost-constrained search strategy configuration",
    )

    # Sweep parameters
    noise_sigmas: list[float] = Field(
        description="Noise levels to sweep over",
    )

    num_seeds: int = Field(
        description="Number of random seeds for statistical significance",
        gt=0,
    )
