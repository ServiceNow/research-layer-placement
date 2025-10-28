"""Convergence analysis command."""

import logging
from pathlib import Path
from typing import ClassVar, Literal

from hydra.core.hydra_config import HydraConfig
import matplotlib.pyplot as plt
import numpy as np
from pydantic import Field
from scipy import stats
import seaborn as sns

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.evaluator import EvaluatorConfig
from place_layers.evaluator.base_evaluator import BaseEvaluator
from place_layers.problem import ProblemConfig
from place_layers.problem.layer_placement import LayerPlacement
from place_layers.search_strategy import SearchStrategyConfig


logger = logging.getLogger(__name__)


class TrackingEvaluatorWrapper(BaseEvaluator[LayerPlacement]):
    """Wrapper that tracks evaluation history for convergence analysis."""

    def __init__(self, wrapped: BaseEvaluator[LayerPlacement]) -> None:
        # Copy config from wrapped evaluator
        super().__init__(wrapped.config)
        self.wrapped = wrapped
        self.evaluation_history: list[tuple[LayerPlacement, float]] = []
        self.eval_count = 0

    async def evaluate(self, parameters: LayerPlacement) -> float:
        """Evaluate and track."""
        score = await self.wrapped.evaluate(parameters)
        self.eval_count += 1
        self.evaluation_history.append((parameters, score))
        return score

    async def evaluate_batch(
        self, parameters_list: list[LayerPlacement]
    ) -> list[float]:
        """Evaluate batch and track."""
        scores = await self.wrapped.evaluate_batch(parameters_list)
        for params, score in zip(parameters_list, scores, strict=False):
            self.eval_count += 1
            self.evaluation_history.append((params, score))
        return scores

    def reset(self) -> None:
        """Reset tracking for new run."""
        self.evaluation_history = []
        self.eval_count = 0

    def get_convergence_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """Get convergence curve: (eval_counts, best_scores_so_far)."""
        if not self.evaluation_history:
            return np.array([]), np.array([])

        eval_counts = np.arange(1, len(self.evaluation_history) + 1)
        scores = np.array([score for _, score in self.evaluation_history])

        # Compute running maximum
        best_so_far = np.maximum.accumulate(scores)

        return eval_counts, best_so_far


class ConvergenceAnalysisCommand(BaseCommand):
    """Analyze convergence behavior of beam search across multiple runs."""

    config: "ConvergenceAnalysisCommandConfig"

    async def run(self) -> None:
        """Execute convergence analysis."""
        # Get output directory from Hydra's runtime config
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)

        n_runs = self.config.n_runs

        # Get base configs
        base_evaluator_config = self.config.evaluator
        base_seed = base_evaluator_config.seed

        logger.info(f"\n{'='*60}")
        logger.info("CONVERGENCE ANALYSIS")
        logger.info(f"{'='*60}")
        logger.info(f"Number of runs: {n_runs}")
        logger.info(
            f"Base seed: {base_seed} (varying from {base_seed} to {base_seed + n_runs - 1})"
        )
        problem_config = self.config.problem

        # Storage for convergence curves
        all_curves = []
        all_final_scores = []
        all_total_evals = []

        # Run multiple times with different evaluator seeds
        for run_idx in range(n_runs):
            run_seed = base_seed + run_idx
            logger.info(f"\nRun {run_idx + 1}/{n_runs} (seed={run_seed})...")

            # Create evaluator with different seed for this run
            evaluator_config = base_evaluator_config.model_copy(
                update={"seed": run_seed}
            )
            evaluator = evaluator_config.create_evaluator()

            # Create tracking wrapper
            tracking_evaluator = TrackingEvaluatorWrapper(evaluator)

            # Create problem with tracking evaluator
            problem = problem_config.create_problem(tracking_evaluator)

            # Create search strategy
            search_strategy = self.config.search_strategy.create_search_strategy()

            # Run search
            result = await search_strategy.search(problem)

            # Extract convergence curve
            eval_counts, best_scores = tracking_evaluator.get_convergence_curve()

            all_curves.append((eval_counts, best_scores))
            all_final_scores.append(result.best_score)
            all_total_evals.append(result.total_evaluations)

            logger.info(f"  Final score: {result.best_score:.4f}")
            logger.info(f"  Total evaluations: {result.total_evaluations:,}")

        # Statistics
        final_scores = np.array(all_final_scores)
        total_evals = np.array(all_total_evals)

        logger.info(f"\n{'='*60}")
        logger.info("CONVERGENCE STATISTICS")
        logger.info(f"{'='*60}")
        logger.info("\nFinal Scores:")
        logger.info(f"  Mean:   {final_scores.mean():.4f}")
        logger.info(f"  Std:    {final_scores.std():.4f}")
        logger.info(f"  Min:    {final_scores.min():.4f}")
        logger.info(f"  Max:    {final_scores.max():.4f}")
        logger.info(f"  Range:  {final_scores.max() - final_scores.min():.4f}")

        logger.info("\nEvaluation Counts:")
        logger.info(f"  Mean:   {total_evals.mean():.1f}")
        logger.info(f"  Std:    {total_evals.std():.1f}")
        logger.info(f"  Min:    {total_evals.min()}")
        logger.info(f"  Max:    {total_evals.max()}")

        # Create plots
        self._create_plots(all_curves, final_scores, total_evals, n_runs, output_dir)
        self._save_statistics(final_scores, total_evals, all_curves, n_runs, output_dir)

        logger.info(f"\n{'='*60}")
        logger.info("ANALYSIS COMPLETE")
        logger.info(f"{'='*60}\n")

    def _create_plots(
        self,
        all_curves: list[tuple[np.ndarray, np.ndarray]],
        final_scores: np.ndarray,
        total_evals: np.ndarray,
        n_runs: int,
        output_dir: Path,
    ) -> None:
        """Create convergence analysis plots."""
        sns.set_style("whitegrid")
        sns.set_palette("husl")

        # Plot 1: All convergence curves overlaid
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Individual curves
        ax = axes[0, 0]
        for _i, (eval_counts, best_scores) in enumerate(all_curves):
            ax.plot(eval_counts, best_scores, alpha=0.3, linewidth=1)
        ax.set_xlabel("Evaluations", fontsize=12)
        ax.set_ylabel("Best Score Found", fontsize=12)
        ax.set_title(f"Convergence Curves (n={n_runs})", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Mean curve with confidence bands
        ax = axes[0, 1]

        # Interpolate all curves to common evaluation points
        max_evals = max(len(curve[0]) for curve in all_curves)
        common_evals = np.linspace(1, max_evals, 200)

        interpolated_scores = []
        for eval_counts, best_scores in all_curves:
            # Interpolate this curve
            interp_scores = np.interp(common_evals, eval_counts, best_scores)
            interpolated_scores.append(interp_scores)

        interpolated_scores = np.array(interpolated_scores)
        mean_curve = interpolated_scores.mean(axis=0)
        std_curve = interpolated_scores.std(axis=0)

        ax.plot(common_evals, mean_curve, linewidth=2, label="Mean", color="blue")
        ax.fill_between(
            common_evals,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.3,
            color="blue",
            label="±1 std",
        )
        ax.fill_between(
            common_evals,
            mean_curve - 2 * std_curve,
            mean_curve + 2 * std_curve,
            alpha=0.15,
            color="blue",
            label="±2 std",
        )
        ax.set_xlabel("Evaluations", fontsize=12)
        ax.set_ylabel("Best Score Found", fontsize=12)
        ax.set_title(
            "Mean Convergence with Confidence Bands", fontsize=14, fontweight="bold"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Final score distribution
        ax = axes[1, 0]
        ax.hist(
            final_scores,
            bins=20,
            density=True,
            alpha=0.6,
            color="skyblue",
            edgecolor="black",
        )
        if final_scores.std() > 0:
            kde = stats.gaussian_kde(final_scores)
            x_range = np.linspace(final_scores.min(), final_scores.max(), 100)
            ax.plot(x_range, kde(x_range), "r-", linewidth=2, label="KDE")
        ax.axvline(
            final_scores.mean(),
            color="green",
            linestyle="--",
            linewidth=2,
            label="Mean",
        )
        ax.set_xlabel("Final Score", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Final Score Distribution", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Evaluation count distribution
        ax = axes[1, 1]
        ax.hist(total_evals, bins=20, alpha=0.6, color="lightcoral", edgecolor="black")
        ax.axvline(
            total_evals.mean(), color="green", linestyle="--", linewidth=2, label="Mean"
        )
        ax.set_xlabel("Total Evaluations", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Evaluation Count Distribution", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "convergence_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"\n✓ Saved plot: {plot_path}")

        # Plot 2: Convergence rate analysis
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Score improvement over time
        ax = axes[0]
        # Compute improvement: derivative of mean curve
        improvement = np.diff(mean_curve) / np.diff(common_evals)
        ax.plot(common_evals[:-1], improvement, linewidth=2, color="purple")
        ax.axhline(0, color="black", linestyle="--", alpha=0.3)
        ax.set_xlabel("Evaluations", fontsize=12)
        ax.set_ylabel("Score Improvement Rate", fontsize=12)
        ax.set_title("Convergence Rate Over Time", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Quantile convergence
        ax = axes[1]
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        for q in quantiles:
            quantile_curve = np.percentile(interpolated_scores, q * 100, axis=0)
            ax.plot(
                common_evals,
                quantile_curve,
                linewidth=2,
                label=f"{int(q*100)}th percentile",
            )
        ax.set_xlabel("Evaluations", fontsize=12)
        ax.set_ylabel("Best Score Found", fontsize=12)
        ax.set_title("Convergence by Quantile", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "convergence_rate.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"✓ Saved plot: {plot_path}")

    def _save_statistics(
        self,
        final_scores: np.ndarray,
        total_evals: np.ndarray,
        all_curves: list[tuple[np.ndarray, np.ndarray]],
        n_runs: int,
        output_dir: Path,
    ) -> None:
        """Save statistics to file."""
        # Interpolate for convergence milestones
        max_evals = max(len(curve[0]) for curve in all_curves)
        common_evals = np.linspace(1, max_evals, 200)

        interpolated_scores = []
        for eval_counts, best_scores in all_curves:
            interp_scores = np.interp(common_evals, eval_counts, best_scores)
            interpolated_scores.append(interp_scores)

        interpolated_scores = np.array(interpolated_scores)
        mean_curve = interpolated_scores.mean(axis=0)

        stats_path = output_dir / "statistics.txt"
        with open(stats_path, "w") as f:
            f.write("CONVERGENCE ANALYSIS STATISTICS\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Number of runs: {n_runs}\n\n")

            f.write("Final Scores:\n")
            f.write(f"  Mean:   {final_scores.mean():.6f}\n")
            f.write(f"  Std:    {final_scores.std():.6f}\n")
            f.write(f"  Min:    {final_scores.min():.6f}\n")
            f.write(f"  Max:    {final_scores.max():.6f}\n")
            f.write(f"  Range:  {final_scores.max() - final_scores.min():.6f}\n")
            f.write(f"  CV:     {final_scores.std() / final_scores.mean():.4f}\n\n")

            f.write("Evaluation Counts:\n")
            f.write(f"  Mean:   {total_evals.mean():.1f}\n")
            f.write(f"  Std:    {total_evals.std():.1f}\n")
            f.write(f"  Min:    {total_evals.min()}\n")
            f.write(f"  Max:    {total_evals.max()}\n")
            f.write(f"  CV:     {total_evals.std() / total_evals.mean():.4f}\n\n")

            # Convergence milestones
            f.write("Convergence Milestones (mean curve):\n")
            target_percentages = [50, 75, 90, 95, 99]
            final_mean = mean_curve[-1]
            for pct in target_percentages:
                target_score = pct / 100 * final_mean
                idx = np.searchsorted(mean_curve, target_score)
                if idx < len(common_evals):
                    f.write(f"  {pct}% of final: {common_evals[idx]:.0f} evaluations\n")

        logger.info(f"✓ Saved statistics: {stats_path}")


class ConvergenceAnalysisCommandConfig(BaseCommandConfig):
    """Configuration for convergence analysis command."""

    _command_class: ClassVar[type[BaseCommand]] = ConvergenceAnalysisCommand

    command: Literal["convergence_analysis"] = "convergence_analysis"

    problem: ProblemConfig = Field(description="Problem configuration")

    evaluator: EvaluatorConfig = Field(description="Evaluator configuration")

    search_strategy: SearchStrategyConfig = Field(
        description="Search strategy configuration"
    )

    n_runs: int = Field(
        description="Number of beam search runs",
        gt=0,
    )
