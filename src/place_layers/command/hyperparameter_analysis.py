"""Hyperparameter analysis command for beam search parameters."""

import csv
import logging
from pathlib import Path
from typing import ClassVar, Literal

from hydra.core.hydra_config import HydraConfig
import matplotlib.pyplot as plt
import numpy as np
from pydantic import Field
import seaborn as sns

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.evaluator import EvaluatorConfig
from place_layers.problem import ProblemConfig
from place_layers.search_strategy import SearchStrategyConfig


logger = logging.getLogger(__name__)


class HyperparameterAnalysisCommand(BaseCommand):
    """Analyze beam search hyperparameter sensitivity."""

    config: "HyperparameterAnalysisCommandConfig"

    async def run(self) -> None:
        """Execute hyperparameter analysis."""
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)

        logger.info(f"\n{'='*60}")
        logger.info("HYPERPARAMETER ANALYSIS")
        logger.info(f"{'='*60}")
        logger.info(f"Beam widths to test: {self.config.beam_widths}")
        logger.info(
            f"Early stop thresholds to test: {self.config.early_stop_thresholds}"
        )

        # Calculate total combinations
        total_runs = len(self.config.beam_widths) * len(
            self.config.early_stop_thresholds
        )
        logger.info(f"Total configurations: {total_runs}")

        # Storage for results
        results = []

        # Run grid search with shared evaluator
        async with self.config.evaluator.create_evaluator() as evaluator:
            problem = self.config.problem.create_problem(evaluator)

            run_idx = 0
            for beam_width in self.config.beam_widths:
                for early_stop_threshold in self.config.early_stop_thresholds:
                    run_idx += 1
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Configuration {run_idx}/{total_runs}")
                    logger.info(f"  beam_width: {beam_width}")
                    logger.info(f"  early_stop_threshold: {early_stop_threshold}")
                    logger.info(f"{'='*60}")

                    # Create modified search strategy config
                    search_config = self.config.search_strategy.model_copy(
                        update={
                            "beam_width": beam_width,
                            "initial_beam_width": beam_width,
                            "early_stop_threshold": early_stop_threshold,
                        }
                    )

                    search_strategy = search_config.create_search_strategy()

                    # Run search
                    result = await search_strategy.search(problem)

                    # Store results
                    results.append(
                    {
                        "beam_width": beam_width,
                        "early_stop_threshold": early_stop_threshold,
                        "final_score": result.best_score,
                        "total_evaluations": result.total_evaluations,
                    }
                )

                logger.info("\nResult:")
                logger.info(f"  Final score: {result.best_score:.4f}")
                logger.info(f"  Total evaluations: {result.total_evaluations}")

        # Save results to CSV
        csv_path = output_dir / "hyperparameter_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "beam_width",
                    "early_stop_threshold",
                    "final_score",
                    "total_evaluations",
                ],
            )
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"\nResults saved to: {csv_path}")

        # Generate visualizations
        self._plot_pareto_frontier(results, output_dir)
        self._plot_heatmaps(results, output_dir)
        self._write_summary(results, output_dir)

        logger.info(f"\n{'='*60}")
        logger.info("HYPERPARAMETER ANALYSIS COMPLETE")
        logger.info(f"{'='*60}")

    def _plot_pareto_frontier(self, results: list[dict], output_dir: Path) -> None:
        """Plot Pareto frontier of evaluations vs score."""
        logger.info("\nGenerating Pareto frontier plot...")

        beam_widths = np.array([r["beam_width"] for r in results])
        evals = np.array([r["total_evaluations"] for r in results])
        scores = np.array([r["final_score"] for r in results])

        fig, ax = plt.subplots(figsize=(10, 6))

        # Color by beam width
        unique_widths = sorted(set(beam_widths))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_widths)))
        color_map = dict(zip(unique_widths, colors, strict=False))

        for width in unique_widths:
            mask = beam_widths == width
            ax.scatter(
                evals[mask],
                scores[mask],
                c=[color_map[width]],
                label=f"width={width}",
                s=100,
                alpha=0.7,
                edgecolors="black",
                linewidth=1,
            )

        ax.set_xlabel("Total Evaluations", fontsize=12)
        ax.set_ylabel("Final Score", fontsize=12)
        ax.set_title(
            "Hyperparameter Trade-off: Evaluations vs Score",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(title="Beam Width", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "pareto_frontier.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Pareto frontier plot saved to: {plot_path}")

    def _plot_heatmaps(self, results: list[dict], output_dir: Path) -> None:
        """Plot heatmaps showing parameter effects."""
        logger.info("\nGenerating heatmaps...")

        # Organize data into matrices
        beam_widths = sorted({r["beam_width"] for r in results})
        thresholds = sorted({r["early_stop_threshold"] for r in results})

        score_matrix = np.zeros((len(thresholds), len(beam_widths)))
        eval_matrix = np.zeros((len(thresholds), len(beam_widths)))

        for r in results:
            i = thresholds.index(r["early_stop_threshold"])
            j = beam_widths.index(r["beam_width"])
            score_matrix[i, j] = r["final_score"]
            eval_matrix[i, j] = r["total_evaluations"]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Score heatmap
        sns.heatmap(
            score_matrix,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            xticklabels=beam_widths,
            yticklabels=[f"{t:.4f}" for t in thresholds],
            ax=ax1,
            cbar_kws={"label": "Final Score"},
        )
        ax1.set_xlabel("Beam Width", fontsize=12)
        ax1.set_ylabel("Early Stop Threshold", fontsize=12)
        ax1.set_title("Final Score by Hyperparameters", fontsize=14, fontweight="bold")

        # Evaluation count heatmap
        sns.heatmap(
            eval_matrix,
            annot=True,
            fmt=".0f",
            cmap="YlOrRd",
            xticklabels=beam_widths,
            yticklabels=[f"{t:.4f}" for t in thresholds],
            ax=ax2,
            cbar_kws={"label": "Total Evaluations"},
        )
        ax2.set_xlabel("Beam Width", fontsize=12)
        ax2.set_ylabel("Early Stop Threshold", fontsize=12)
        ax2.set_title(
            "Evaluation Count by Hyperparameters", fontsize=14, fontweight="bold"
        )

        plt.tight_layout()
        plot_path = output_dir / "hyperparameter_heatmaps.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Heatmaps saved to: {plot_path}")

    def _write_summary(self, results: list[dict], output_dir: Path) -> None:
        """Write summary statistics to file."""
        logger.info("\nGenerating summary statistics...")

        summary_path = output_dir / "summary.txt"
        with open(summary_path, "w") as f:
            f.write("HYPERPARAMETER ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            # Best configurations
            best_score = max(results, key=lambda r: r["final_score"])
            fewest_evals = min(results, key=lambda r: r["total_evaluations"])

            f.write("Best Score Configuration:\n")
            f.write(f"  beam_width: {best_score['beam_width']}\n")
            f.write(f"  early_stop_threshold: {best_score['early_stop_threshold']}\n")
            f.write(f"  final_score: {best_score['final_score']:.4f}\n")
            f.write(f"  total_evaluations: {best_score['total_evaluations']}\n\n")

            f.write("Fewest Evaluations Configuration:\n")
            f.write(f"  beam_width: {fewest_evals['beam_width']}\n")
            f.write(f"  early_stop_threshold: {fewest_evals['early_stop_threshold']}\n")
            f.write(f"  final_score: {fewest_evals['final_score']:.4f}\n")
            f.write(f"  total_evaluations: {fewest_evals['total_evaluations']}\n\n")

            # Score at fewest evals relative to best
            score_gap = best_score["final_score"] - fewest_evals["final_score"]
            score_gap_pct = (
                100 * score_gap / abs(best_score["final_score"])
                if best_score["final_score"] != 0
                else 0
            )
            eval_reduction = 100 * (
                1 - fewest_evals["total_evaluations"] / best_score["total_evaluations"]
            )

            f.write("Trade-off Analysis:\n")
            f.write(f"  Score gap: {score_gap:.4f} ({score_gap_pct:.2f}%)\n")
            f.write(f"  Evaluation reduction: {eval_reduction:.1f}%\n\n")

            # Parameter effects
            f.write("Parameter Ranges:\n")
            beam_widths = sorted({r["beam_width"] for r in results})
            thresholds = sorted({r["early_stop_threshold"] for r in results})
            f.write(f"  beam_widths tested: {beam_widths}\n")
            f.write(f"  early_stop_thresholds tested: {thresholds}\n\n")

            # Score statistics by beam width
            f.write("Score Statistics by Beam Width:\n")
            for width in beam_widths:
                width_results = [r for r in results if r["beam_width"] == width]
                scores = [r["final_score"] for r in width_results]
                f.write(
                    f"  width={width}: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}, "
                    f"min={np.min(scores):.4f}, max={np.max(scores):.4f}\n"
                )

        logger.info(f"Summary saved to: {summary_path}")


class HyperparameterAnalysisCommandConfig(BaseCommandConfig):
    """Configuration for hyperparameter analysis command."""

    _command_class: ClassVar[type[BaseCommand]] = HyperparameterAnalysisCommand

    command: Literal["hyperparameter_analysis"] = "hyperparameter_analysis"

    # Fixed landscape
    evaluator: EvaluatorConfig = Field(
        description="Evaluator configuration (fixed seed for reproducibility)"
    )
    problem: ProblemConfig = Field(description="Problem configuration")
    search_strategy: SearchStrategyConfig = Field(
        description="Base search strategy configuration (will be modified with test parameters)"
    )

    # Parameter ranges to test
    beam_widths: list[int] = Field(
        description="List of beam widths to test",
    )
    early_stop_thresholds: list[float] = Field(
        description="List of early stopping thresholds to test",
    )
