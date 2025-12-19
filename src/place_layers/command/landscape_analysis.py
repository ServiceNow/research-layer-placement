"""Landscape analysis command."""

from itertools import combinations
import logging
from math import comb
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
from place_layers.core.types import LayerPlacement
from place_layers.evaluator import EvaluatorConfig
from place_layers.problem import ProblemConfig
from place_layers.problem.layer_placement import LayerPlacementProblemConfig
from place_layers.search_strategy import SearchStrategyConfig


logger = logging.getLogger(__name__)


class LandscapeAnalysisCommand(BaseCommand):
    """Analyze the score landscape of a layer placement problem."""

    config: "LandscapeAnalysisCommandConfig"

    async def run(self) -> None:
        """Execute landscape analysis."""
        # Get output directory from Hydra's runtime config
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)

        # Create problem
        evaluator = self.config.evaluator.create_evaluator()
        problem = self.config.problem.create_problem(evaluator)

        if not isinstance(problem.config, LayerPlacementProblemConfig):
            raise TypeError("This command only works with LayerPlacementProblem")

        num_layers = evaluator.num_layers
        num_mixers = evaluator.num_mixers
        budgets = problem.config.budgets

        logger.info(f"\n{'='*60}")
        logger.info("LANDSCAPE ANALYSIS")
        logger.info(f"{'='*60}")
        logger.info(f"Layers: {num_layers}")
        logger.info(f"Mixers: {num_mixers}")
        logger.info(f"Budgets: {budgets}")

        # Estimate search space
        search_space_size = self._estimate_search_space(num_layers, budgets)
        logger.info(f"Total configurations: {search_space_size:,}")

        # Generate or sample configurations
        sample_size = self.config.sample_size
        is_exhaustive = False
        if sample_size is None:
            if search_space_size > 100_000:
                logger.warning("Search space too large, using sample_size=10,000")
                sample_size = 10_000
            else:
                logger.info(f"Generating all {search_space_size:,} configurations...")
                configs = self._generate_all_configs(num_layers, num_mixers, budgets)
                is_exhaustive = True

        if sample_size is not None:
            logger.info(f"Sampling {sample_size:,} random configurations...")
            configs = self._sample_configs(num_layers, num_mixers, budgets, sample_size)

        # Evaluate all configs
        logger.info(f"Evaluating {len(configs):,} configurations...")
        scores = await problem.evaluate_batch(configs)
        scores = np.array(scores)

        # Statistics
        logger.info(f"\n{'='*60}")
        logger.info("SCORE DISTRIBUTION STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Min:     {scores.min():.4f}")
        logger.info(f"Q1:      {np.percentile(scores, 25):.4f}")
        logger.info(f"Median:  {np.median(scores):.4f}")
        logger.info(f"Q3:      {np.percentile(scores, 75):.4f}")
        logger.info(f"Max:     {scores.max():.4f}")
        logger.info(f"Mean:    {scores.mean():.4f}")
        logger.info(f"Std:     {scores.std():.4f}")
        logger.info(f"Range:   {scores.max() - scores.min():.4f}")

        # Near-optimal analysis
        best_score = scores.max()
        thresholds = [0.99, 0.95, 0.90]
        best_label = "optimal" if is_exhaustive else "best in sample"
        logger.info("\nNear-optimal configurations:")
        for thresh in thresholds:
            count = (scores >= thresh * best_score).sum()
            pct = 100 * count / len(scores)
            logger.info(
                f"  Within {100*(1-thresh):.0f}% of {best_label}: {count:,} ({pct:.2f}%)"
            )

        # Run beam search for comparison
        logger.info(f"\n{'='*60}")
        logger.info("RUNNING BEAM SEARCH")
        logger.info(f"{'='*60}")
        search_strategy = self.config.search_strategy.create_search_strategy()
        result = await search_strategy.search(problem)
        beam_search_score = result.best_score

        comparison_label = "Global optimum" if is_exhaustive else "Best in sample"
        gap = beam_search_score - best_score
        gap_interpretation = "better" if gap > 0 else "worse"

        logger.info(f"Beam search found: {beam_search_score:.4f}")
        logger.info(f"{comparison_label}: {best_score:.4f}")
        logger.info(
            f"Gap (BS - {comparison_label[:4]}): {gap:+.4f} ({abs(gap):.4f} {gap_interpretation})"
        )

        # Percentile: clamp to 100% if beam search exceeds sample
        if beam_search_score > best_score:
            percentile_str = ">100.00% (exceeds sample)"
        else:
            percentile = (scores <= beam_search_score).sum() / len(scores) * 100
            percentile_str = f"{percentile:.2f}%"
        logger.info(f"Percentile:        {percentile_str}")

        # Create plots
        self._create_plots(
            scores, best_score, beam_search_score, is_exhaustive, output_dir
        )
        self._save_statistics(
            scores,
            best_score,
            beam_search_score,
            num_layers,
            num_mixers,
            budgets,
            search_space_size,
            len(configs),
            thresholds,
            is_exhaustive,
            output_dir,
        )

        logger.info(f"\n{'='*60}")
        logger.info("ANALYSIS COMPLETE")
        logger.info(f"{'='*60}\n")

    def _estimate_search_space(self, num_layers: int, budgets: list[int]) -> int:
        """Estimate number of valid configurations."""
        size = 1
        remaining = num_layers
        for budget in budgets:
            size *= comb(remaining, budget)
            remaining -= budget
        return size

    def _generate_all_configs(
        self, num_layers: int, num_mixers: int, budgets: list[int]
    ) -> list[LayerPlacement]:
        """Generate all valid configurations."""
        configs = []

        def enumerate_recursive(
            current_assignment: list[int],
            available_layers: list[int],
            phase_idx: int,
        ) -> None:
            if phase_idx >= len(budgets):
                final_assignment = current_assignment.copy()
                for layer in available_layers:
                    final_assignment[layer] = num_mixers - 1
                configs.append(LayerPlacement(assignments=tuple(final_assignment)))
                return

            budget = budgets[phase_idx]
            mixer_idx = phase_idx

            for selected_layers in combinations(available_layers, budget):
                new_assignment = current_assignment.copy()
                for layer in selected_layers:
                    new_assignment[layer] = mixer_idx
                new_available = [
                    layer for layer in available_layers if layer not in selected_layers
                ]
                enumerate_recursive(new_assignment, new_available, phase_idx + 1)

        initial_assignment = [-1] * num_layers
        initial_available = list(range(num_layers))
        enumerate_recursive(initial_assignment, initial_available, 0)

        return configs

    def _sample_configs(
        self,
        num_layers: int,
        num_mixers: int,
        budgets: list[int],
        sample_size: int,
        seed: int = 42,
    ) -> list[LayerPlacement]:
        """Sample random valid configurations."""
        import random

        rng = random.Random(seed)
        configs = []

        for _ in range(sample_size):
            assignment = [-1] * num_layers
            available = list(range(num_layers))

            for phase_idx, budget in enumerate(budgets):
                selected = rng.sample(available, budget)
                for layer in selected:
                    assignment[layer] = phase_idx
                    available.remove(layer)

            for layer in available:
                assignment[layer] = num_mixers - 1

            configs.append(LayerPlacement(assignments=tuple(assignment)))

        return configs

    def _create_plots(
        self,
        scores: np.ndarray,
        best_score: float,
        beam_search_score: float,
        is_exhaustive: bool,
        output_dir: Path,
    ) -> None:
        """Create landscape analysis plots."""
        sns.set_style("whitegrid")
        sns.set_palette("husl")

        # Label depends on whether we did exhaustive search
        best_label = "Global Optimum" if is_exhaustive else "Best in Sample"

        # Plot 1: Distribution analysis
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Histogram + KDE
        ax = axes[0, 0]
        ax.hist(
            scores, bins=50, density=True, alpha=0.6, color="skyblue", edgecolor="black"
        )
        kde = stats.gaussian_kde(scores)
        x_range = np.linspace(scores.min(), scores.max(), 200)
        ax.plot(x_range, kde(x_range), "r-", linewidth=2, label="KDE")
        ax.axvline(
            beam_search_score,
            color="green",
            linestyle="--",
            linewidth=2,
            label="Beam Search",
        )
        ax.axvline(
            best_score, color="orange", linestyle="--", linewidth=2, label=best_label
        )
        ax.set_xlabel("Score", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Score Distribution", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Box plot
        ax = axes[0, 1]
        bp = ax.boxplot([scores], orientation="vertical", patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("lightblue")
        ax.plot([1], [beam_search_score], "go", markersize=10, label="Beam Search")
        ax.plot([1], [best_score], "o", color="orange", markersize=10, label=best_label)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Box Plot", fontsize=14, fontweight="bold")
        ax.set_xticklabels(["All Configs"])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ECDF
        ax = axes[1, 0]
        sorted_scores = np.sort(scores)
        ecdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        ax.plot(sorted_scores, ecdf, linewidth=2)
        percentile = (scores <= beam_search_score).sum() / len(scores)
        ax.axvline(beam_search_score, color="green", linestyle="--", linewidth=2)
        ax.axhline(percentile, color="green", linestyle=":", alpha=0.5)
        ax.plot(
            [beam_search_score],
            [percentile],
            "go",
            markersize=10,
            label=f"Beam Search ({percentile*100:.1f}%)",
        )
        ax.axvline(
            best_score, color="orange", linestyle="--", linewidth=2, label=best_label
        )
        ax.set_xlabel("Score", fontsize=12)
        ax.set_ylabel("Cumulative Probability", fontsize=12)
        ax.set_title("Empirical CDF", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Q-Q plot
        ax = axes[1, 1]
        stats.probplot(scores, dist="norm", plot=ax)
        ax.set_xlabel("Theoretical Quantiles", fontsize=12)
        ax.set_ylabel("Sample Quantiles", fontsize=12)
        ax.set_title("Q-Q Plot (vs Normal)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        # probplot creates a line plot - get the line and add legend
        lines = ax.get_lines()
        if len(lines) >= 2:
            ax.legend([lines[0], lines[1]], ["Sample Data", "Normal Fit"], fontsize=10)

        plt.tight_layout()
        plot_path = output_dir / "landscape_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"\n✓ Saved plot: {plot_path}")

        # Plot 2: Percentile analysis
        fig, ax = plt.subplots(figsize=(10, 6))
        percentiles = np.arange(0, 101, 1)
        percentile_values = np.percentile(scores, percentiles)
        ax.plot(percentiles, percentile_values, linewidth=2)
        ax.fill_between(percentiles, percentile_values, alpha=0.3)

        beam_percentile = (scores <= beam_search_score).sum() / len(scores) * 100
        ax.axhline(beam_search_score, color="green", linestyle="--", linewidth=2)
        ax.axvline(beam_percentile, color="green", linestyle="--", linewidth=2)
        ax.plot(
            [beam_percentile],
            [beam_search_score],
            "go",
            markersize=12,
            label=f"Beam Search ({beam_percentile:.1f}th percentile)",
        )

        ax.axhline(
            best_score,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"{best_label} (100th)",
        )
        ax.set_xlabel("Percentile", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Score by Percentile", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "percentile_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"✓ Saved plot: {plot_path}")

    def _save_statistics(
        self,
        scores: np.ndarray,
        best_score: float,
        beam_search_score: float,
        num_layers: int,
        num_mixers: int,
        budgets: list[int],
        search_space_size: int,
        evaluated: int,
        thresholds: list[float],
        is_exhaustive: bool,
        output_dir: Path,
    ) -> None:
        """Save statistics to file."""
        comparison_label = "Global optimum" if is_exhaustive else "Best in sample"
        best_label = "optimal" if is_exhaustive else "best in sample"
        gap = beam_search_score - best_score

        # Percentile handling
        if beam_search_score > best_score:
            percentile_str = ">100.00% (exceeds sample)"
        else:
            beam_percentile = (scores <= beam_search_score).sum() / len(scores) * 100
            percentile_str = f"{beam_percentile:.2f}%"

        stats_path = output_dir / "statistics.txt"
        with open(stats_path, "w") as f:
            f.write("LANDSCAPE ANALYSIS STATISTICS\n")
            f.write(f"{'='*60}\n\n")
            f.write("Configuration:\n")
            f.write(f"  Layers: {num_layers}\n")
            f.write(f"  Mixers: {num_mixers}\n")
            f.write(f"  Budgets: {budgets}\n")
            f.write(f"  Total configs: {search_space_size:,}\n")
            f.write(f"  Evaluated: {evaluated:,}\n")
            f.write(f"  Exhaustive: {'Yes' if is_exhaustive else 'No (sampled)'}\n\n")
            f.write("Distribution:\n")
            f.write(f"  Min:     {scores.min():.6f}\n")
            f.write(f"  Q1:      {np.percentile(scores, 25):.6f}\n")
            f.write(f"  Median:  {np.median(scores):.6f}\n")
            f.write(f"  Q3:      {np.percentile(scores, 75):.6f}\n")
            f.write(f"  Max:     {scores.max():.6f}\n")
            f.write(f"  Mean:    {scores.mean():.6f}\n")
            f.write(f"  Std:     {scores.std():.6f}\n")
            f.write(f"  Range:   {scores.max() - scores.min():.6f}\n\n")

            f.write("Beam Search Comparison:\n")
            f.write(f"  Beam search score: {beam_search_score:.6f}\n")
            f.write(f"  {comparison_label}:  {best_score:.6f}\n")
            f.write(f"  Gap (BS - {comparison_label[:4]}): {gap:+.6f}\n")
            f.write(f"  Percentile:        {percentile_str}\n\n")

            f.write(f"Near-{best_label} configs:\n")
            for thresh in thresholds:
                count = (scores >= thresh * best_score).sum()
                pct = 100 * count / len(scores)
                f.write(f"  Within {100*(1-thresh):.0f}%: {count:,} ({pct:.2f}%)\n")

        logger.info(f"✓ Saved statistics: {stats_path}")


class LandscapeAnalysisCommandConfig(BaseCommandConfig):
    """Configuration for landscape analysis command."""

    _command_class: ClassVar[type[BaseCommand]] = LandscapeAnalysisCommand

    command: Literal["landscape_analysis"] = "landscape_analysis"

    problem: ProblemConfig = Field(description="Problem configuration")

    evaluator: EvaluatorConfig = Field(description="Evaluator configuration")

    search_strategy: SearchStrategyConfig = Field(
        description="Search strategy configuration for comparison"
    )

    sample_size: int | None = Field(
        description="Number of configs to sample (null = exhaustive if < 100k)",
    )
