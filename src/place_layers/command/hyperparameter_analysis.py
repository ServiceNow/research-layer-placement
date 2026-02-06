"""Hyperparameter analysis command for beam search parameters."""

import csv
import logging
from pathlib import Path
from typing import ClassVar, Literal

from hydra.core.hydra_config import HydraConfig
import matplotlib.pyplot as plt
import numpy as np
from pydantic import Field

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

        # Calculate total combinations
        total_runs = len(self.config.beam_widths)
        logger.info(f"Total configurations: {total_runs}")

        # Storage for results
        results = []

        # Run grid search with shared evaluator
        async with self.config.evaluator.create_evaluator() as evaluator:
            problem = self.config.problem.create_problem(evaluator)

            for run_idx, beam_width in enumerate(self.config.beam_widths, 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"Configuration {run_idx}/{total_runs}")
                logger.info(f"  beam_width: {beam_width}")
                logger.info(f"{'='*60}")

                # Create modified search strategy config
                search_config = self.config.search_strategy.model_copy(
                    update={
                        "beam_width": beam_width,
                        "initial_beam_width": beam_width,
                    }
                )

                search_strategy = search_config.create_search_strategy()

                # Run search
                result = await search_strategy.search(problem)

                # Store results
                results.append(
                    {
                        "beam_width": beam_width,
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
                    "final_score",
                    "total_evaluations",
                ],
            )
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"\nResults saved to: {csv_path}")

        # Generate visualizations
        self._plot_results(results, output_dir)
        self._write_summary(results, output_dir)

        logger.info(f"\n{'='*60}")
        logger.info("HYPERPARAMETER ANALYSIS COMPLETE")
        logger.info(f"{'='*60}")

    def _plot_results(self, results: list[dict], output_dir: Path) -> None:
        """Plot beam width effects on score and evaluations."""
        logger.info("\nGenerating results plots...")

        beam_widths = np.array([r["beam_width"] for r in results])
        evals = np.array([r["total_evaluations"] for r in results])
        scores = np.array([r["final_score"] for r in results])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Score vs beam width
        ax1.plot(beam_widths, scores, "o-", markersize=8, linewidth=2)
        ax1.set_xlabel("Beam Width", fontsize=12)
        ax1.set_ylabel("Final Score", fontsize=12)
        ax1.set_title("Score vs Beam Width", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Evaluations vs beam width
        ax2.plot(beam_widths, evals, "o-", markersize=8, linewidth=2, color="orange")
        ax2.set_xlabel("Beam Width", fontsize=12)
        ax2.set_ylabel("Total Evaluations", fontsize=12)
        ax2.set_title("Evaluations vs Beam Width", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "beam_width_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Results plot saved to: {plot_path}")

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
            f.write(f"  final_score: {best_score['final_score']:.4f}\n")
            f.write(f"  total_evaluations: {best_score['total_evaluations']}\n\n")

            f.write("Fewest Evaluations Configuration:\n")
            f.write(f"  beam_width: {fewest_evals['beam_width']}\n")
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
            f.write("Beam Widths Tested:\n")
            beam_widths = sorted({r["beam_width"] for r in results})
            f.write(f"  {beam_widths}\n\n")

            # Results by beam width
            f.write("Results by Beam Width:\n")
            for width in beam_widths:
                width_result = next(r for r in results if r["beam_width"] == width)
                f.write(
                    f"  width={width}: score={width_result['final_score']:.4f}, "
                    f"evals={width_result['total_evaluations']}\n"
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
