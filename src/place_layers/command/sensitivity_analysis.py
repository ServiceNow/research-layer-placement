"""Sensitivity analysis command."""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, ClassVar, Literal

from hydra.core.hydra_config import HydraConfig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import Field
import seaborn as sns

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.command.convergence_analysis import TrackingEvaluatorWrapper
from place_layers.evaluator import EvaluatorConfig
from place_layers.evaluator.mock_supernet_evaluator import (
    MockSupernetEvaluator,
    MockSupernetEvaluatorConfig,
)
from place_layers.problem import ProblemConfig
from place_layers.search_strategy import SearchStrategyConfig


logger = logging.getLogger(__name__)


@dataclass
class ConvergenceMetrics:
    """Metrics for a single beam search run."""

    final_score: float
    total_evals: int
    evals_to_90pct: int  # Evals to reach 90% of final score
    evals_to_95pct: int  # Evals to reach 95% of final score
    evals_to_99pct: int  # Evals to reach 99% of final score


class SensitivityAnalysisCommand(BaseCommand):
    """Analyze sensitivity of beam search convergence to mock evaluator parameters."""

    config: "SensitivityAnalysisCommandConfig"

    async def run(self) -> None:
        """Execute sensitivity analysis."""
        # Get output directory from Hydra's runtime config
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)

        n_trials_per_config = self.config.n_trials_per_config

        logger.info(f"\n{'='*60}")
        logger.info("PARAMETER SENSITIVITY ANALYSIS")
        logger.info(f"{'='*60}")
        logger.info(f"Trials per configuration: {n_trials_per_config}")

        # Validate evaluator type
        if not isinstance(self.config.evaluator, MockSupernetEvaluatorConfig):
            raise TypeError("This command only works with MockSupernetEvaluator")

        # Define parameter sweep
        param_sweep = {
            "noise_sigma": [0.05, 0.1, 0.2, 0.5, 1.0],
            "pairwise_scale": [0.0, 0.1, 0.3, 0.5, 0.7],
            "unary_bonus": [0.5, 1.0, 1.5, 2.0],
            "unary_penalty": [0.25, 0.5, 0.75, 1.0],
            "seed": [42, 123, 456, 789, 1011],
        }

        logger.info("\nParameter ranges:")
        for param, values in param_sweep.items():
            logger.info(f"  {param}: {values}")

        # Baseline config
        baseline_params = {
            "noise_sigma": self.config.evaluator.noise_sigma,
            "pairwise_scale": self.config.evaluator.pairwise_scale,
            "unary_bonus": self.config.evaluator.unary_bonus,
            "unary_penalty": self.config.evaluator.unary_penalty,
            "seed": self.config.evaluator.seed,
        }

        logger.info("\nBaseline configuration:")
        for param, value in baseline_params.items():
            logger.info(f"  {param}: {value}")

        # Storage for results
        results = []

        # Sweep each parameter independently
        total_trials = (
            sum(len(values) for values in param_sweep.values()) * n_trials_per_config
        )
        trial_count = 0

        for param_name, param_values in param_sweep.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"SWEEPING: {param_name}")
            logger.info(f"{'='*60}")

            for param_value in param_values:
                # Create override (baseline + this parameter)
                overrides = baseline_params.copy()
                overrides[param_name] = param_value

                logger.info(f"\n{param_name}={param_value}:")

                # Run multiple trials
                for trial_idx in range(n_trials_per_config):
                    trial_count += 1
                    logger.info(
                        f"  Trial {trial_idx + 1}/{n_trials_per_config} "
                        f"(overall: {trial_count}/{total_trials})...",
                        end=" ",
                    )

                    metrics = await self._run_single_trial(overrides)

                    # Store result
                    results.append(
                        {
                            "param_name": param_name,
                            "param_value": param_value,
                            "trial": trial_idx,
                            "final_score": metrics.final_score,
                            "total_evals": metrics.total_evals,
                            "evals_to_90pct": metrics.evals_to_90pct,
                            "evals_to_95pct": metrics.evals_to_95pct,
                            "evals_to_99pct": metrics.evals_to_99pct,
                            **{f"config_{k}": v for k, v in overrides.items()},
                        }
                    )

                    logger.info(
                        f"score={metrics.final_score:.3f}, evals={metrics.total_evals}"
                    )

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Compute summary statistics
        self._log_sensitivity_summary(df, param_sweep)

        # Create plots
        self._create_plots(df, param_sweep, output_dir)
        self._save_results(
            df, param_sweep, baseline_params, n_trials_per_config, output_dir
        )

        logger.info(f"\n{'='*60}")
        logger.info("ANALYSIS COMPLETE")
        logger.info(f"{'='*60}\n")

    async def _run_single_trial(
        self, evaluator_overrides: dict[str, Any]
    ) -> ConvergenceMetrics:
        """Run single beam search trial with evaluator parameter overrides."""
        # Create evaluator with overrides
        config_dict = self.config.evaluator.model_dump()
        config_dict.update(evaluator_overrides)
        modified_config = MockSupernetEvaluatorConfig.model_validate(config_dict)

        # Create components
        base_evaluator = MockSupernetEvaluator(modified_config)
        tracking_evaluator = TrackingEvaluatorWrapper(base_evaluator)
        problem = self.config.problem.create_problem(tracking_evaluator)
        search_strategy = self.config.search_strategy.create_search_strategy()

        # Run search
        await search_strategy.search(problem)

        # Extract metrics
        eval_counts, best_scores = tracking_evaluator.get_convergence_curve()
        return self._compute_convergence_metrics(eval_counts, best_scores)

    def _compute_convergence_metrics(
        self, eval_counts: np.ndarray, best_scores: np.ndarray
    ) -> ConvergenceMetrics:
        """Compute convergence metrics from a convergence curve."""
        final_score = best_scores[-1]
        total_evals = len(best_scores)

        # Find evaluations to reach percentages of final score
        def find_eval_to_target(target_score: float) -> int:
            idx = np.searchsorted(best_scores, target_score)
            return min(idx + 1, total_evals)

        evals_to_90pct = find_eval_to_target(0.90 * final_score)
        evals_to_95pct = find_eval_to_target(0.95 * final_score)
        evals_to_99pct = find_eval_to_target(0.99 * final_score)

        return ConvergenceMetrics(
            final_score=final_score,
            total_evals=total_evals,
            evals_to_90pct=evals_to_90pct,
            evals_to_95pct=evals_to_95pct,
            evals_to_99pct=evals_to_99pct,
        )

    def _log_sensitivity_summary(self, df: pd.DataFrame, param_sweep: dict) -> None:
        """Log sensitivity summary statistics."""
        logger.info(f"\n{'='*60}")
        logger.info("SENSITIVITY SUMMARY")
        logger.info(f"{'='*60}\n")

        summary_metrics = ["final_score", "total_evals", "evals_to_95pct"]

        for metric in summary_metrics:
            logger.info(f"\n{metric.upper()}:")
            logger.info(f"{'='*40}")

            # Compute variance explained by each parameter
            total_var = df[metric].var()

            for param_name in param_sweep.keys():
                param_df = df[df["param_name"] == param_name]
                between_group_var = param_df.groupby("param_value")[metric].mean().var()
                variance_explained = (
                    between_group_var / total_var if total_var > 0 else 0
                )

                logger.info(
                    f"  {param_name:20s} explains {variance_explained*100:5.1f}% of variance"
                )

    def _create_plots(
        self, df: pd.DataFrame, param_sweep: dict, output_dir: Path
    ) -> None:
        """Create sensitivity analysis plots."""
        sns.set_style("whitegrid")
        sns.set_palette("husl")

        # Plot 1: Parameter sensitivity for key metrics
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        metrics_to_plot = [
            ("final_score", "Final Score"),
            ("total_evals", "Total Evaluations"),
            ("evals_to_90pct", "Evals to 90%"),
            ("evals_to_95pct", "Evals to 95%"),
            ("evals_to_99pct", "Evals to 99%"),
        ]

        for idx, (metric, label) in enumerate(metrics_to_plot):
            ax = axes[idx]

            for param_name in param_sweep.keys():
                param_df = df[df["param_name"] == param_name]

                # Group by parameter value and compute stats
                grouped = param_df.groupby("param_value")[metric]
                means = grouped.mean()
                stds = grouped.std()

                ax.errorbar(
                    means.index,
                    means.values,
                    yerr=stds.values,
                    marker="o",
                    linewidth=2,
                    capsize=5,
                    label=param_name,
                )

            ax.set_xlabel("Parameter Value", fontsize=11)
            ax.set_ylabel(label, fontsize=11)
            ax.set_title(f"Sensitivity: {label}", fontsize=12, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused subplot
        axes[-1].axis("off")

        plt.tight_layout()
        plot_path = output_dir / "parameter_sensitivity.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"\n✓ Saved plot: {plot_path}")

        # Plot 2: Box plots for each parameter
        fig, axes = plt.subplots(
            len(param_sweep), 3, figsize=(15, 3 * len(param_sweep))
        )

        metrics_for_boxes = [
            ("final_score", "Final Score"),
            ("total_evals", "Total Evals"),
            ("evals_to_95pct", "Evals to 95%"),
        ]

        for row_idx, param_name in enumerate(param_sweep.keys()):
            param_df = df[df["param_name"] == param_name]

            for col_idx, (metric, label) in enumerate(metrics_for_boxes):
                ax = axes[row_idx, col_idx]

                # Create box plot
                param_values = sorted(param_df["param_value"].unique())
                data_to_plot = [
                    param_df[param_df["param_value"] == v][metric].values
                    for v in param_values
                ]

                bp = ax.boxplot(
                    data_to_plot,
                    labels=[str(v) for v in param_values],
                    patch_artist=True,
                )

                for patch in bp["boxes"]:
                    patch.set_facecolor("lightblue")

                ax.set_xlabel(f"{param_name} Value", fontsize=10)
                ax.set_ylabel(label, fontsize=10)
                ax.set_title(f"{param_name} → {label}", fontsize=11, fontweight="bold")
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "parameter_boxplots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"✓ Saved plot: {plot_path}")

        # Plot 3: Variance decomposition
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, (metric, label) in enumerate(metrics_for_boxes):
            ax = axes[idx]

            # Compute variance explained by each parameter
            total_var = df[metric].var()
            variance_explained = {}

            for param_name in param_sweep.keys():
                param_df = df[df["param_name"] == param_name]
                between_group_var = param_df.groupby("param_value")[metric].mean().var()
                variance_explained[param_name] = (
                    between_group_var / total_var if total_var > 0 else 0
                )

            # Bar plot
            params = list(variance_explained.keys())
            variances = [variance_explained[p] * 100 for p in params]

            bars = ax.bar(range(len(params)), variances, color="steelblue", alpha=0.7)
            ax.set_xticks(range(len(params)))
            ax.set_xticklabels(params, rotation=45, ha="right")
            ax.set_ylabel("Variance Explained (%)", fontsize=11)
            ax.set_title(
                f"Parameter Influence on {label}", fontsize=12, fontweight="bold"
            )
            ax.grid(True, alpha=0.3, axis="y")

            # Add value labels on bars
            for _i, (bar, val) in enumerate(zip(bars, variances, strict=False)):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.tight_layout()
        plot_path = output_dir / "variance_decomposition.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"✓ Saved plot: {plot_path}")

    def _save_results(
        self,
        df: pd.DataFrame,
        param_sweep: dict,
        baseline_params: dict,
        n_trials_per_config: int,
        output_dir: Path,
    ) -> None:
        """Save results and statistics to files."""
        # Save results to CSV
        csv_path = output_dir / "sensitivity_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"✓ Saved results: {csv_path}")

        # Save summary statistics
        stats_path = output_dir / "summary.txt"
        summary_metrics = ["final_score", "total_evals", "evals_to_95pct"]

        with open(stats_path, "w") as f:
            f.write("PARAMETER SENSITIVITY ANALYSIS SUMMARY\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Trials per configuration: {n_trials_per_config}\n")
            f.write(f"Total trials: {len(df)}\n\n")

            f.write("Baseline configuration:\n")
            for param, value in baseline_params.items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")

            for metric in summary_metrics:
                f.write(f"\n{metric.upper()} - Variance Explained:\n")
                f.write(f"{'-'*40}\n")

                total_var = df[metric].var()
                for param_name in param_sweep.keys():
                    param_df = df[df["param_name"] == param_name]
                    between_group_var = (
                        param_df.groupby("param_value")[metric].mean().var()
                    )
                    variance_explained = (
                        between_group_var / total_var if total_var > 0 else 0
                    )
                    f.write(f"  {param_name:20s} {variance_explained*100:5.1f}%\n")

            # Calibration recommendations
            f.write("\n\nCALIBRATION RECOMMENDATIONS:\n")
            f.write(f"{'='*60}\n\n")

            # Identify most influential parameters
            influential_params = {}
            for metric in summary_metrics:
                total_var = df[metric].var()
                for param_name in param_sweep.keys():
                    param_df = df[df["param_name"] == param_name]
                    between_group_var = (
                        param_df.groupby("param_value")[metric].mean().var()
                    )
                    variance_explained = (
                        between_group_var / total_var if total_var > 0 else 0
                    )

                    if param_name not in influential_params:
                        influential_params[param_name] = 0
                    influential_params[param_name] += variance_explained

            # Sort by total influence
            sorted_params = sorted(
                influential_params.items(), key=lambda x: x[1], reverse=True
            )

            if sorted_params[0][1] < 0.1:  # Less than 10% variance explained
                f.write(
                    "✓ GOOD NEWS: Convergence behavior is ROBUST to mock parameters!\n"
                )
                f.write("  The mock evaluator appears to be a useful general model.\n")
                f.write("  Algorithm insights should transfer to real supernetworks.\n")
            else:
                f.write("⚠ CALIBRATION NEEDED: Convergence is sensitive to:\n")
                for param, influence in sorted_params[:3]:
                    if influence > 0.1:
                        f.write(
                            f"  - {param} (explains {influence/len(summary_metrics)*100:.1f}% avg variance)\n"
                        )
                f.write(
                    "\nRecommendation: Match these parameters to your real supernet properties.\n"
                )

        logger.info(f"✓ Saved summary: {stats_path}")


class SensitivityAnalysisCommandConfig(BaseCommandConfig):
    """Configuration for sensitivity analysis command."""

    _command_class: ClassVar[type[BaseCommand]] = SensitivityAnalysisCommand

    command: Literal["sensitivity_analysis"] = "sensitivity_analysis"

    problem: ProblemConfig = Field(description="Problem configuration")

    evaluator: EvaluatorConfig = Field(
        description="Evaluator configuration (must be MockSupernetEvaluator)"
    )

    search_strategy: SearchStrategyConfig = Field(
        description="Search strategy configuration"
    )

    n_trials_per_config: int = Field(
        description="Number of trials per parameter configuration",
        gt=0,
    )
