"""Per-problem placement landscape comparison without density estimation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from hydra.core.hydra_config import HydraConfig
from pydantic import BaseModel, Field

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.core.trace_problem_split import PerProblemTraceManifest
from place_layers.landscape import (
    BandResult,
    LandscapeConfig,
    LandscapeData,
    compute_dp_per_band,
    find_objective_result,
)
from place_layers.surrogate.config import (
    BayesianRidgeSurrogateConfig,
)
from place_layers.surrogate.evaluation_log import (
    PooledEvaluationData,
    discover_evaluation_records,
)


logger = logging.getLogger(__name__)


class ProblemLandscapeSummaryItem(BaseModel):
    """Summary for one selected problem in a comparison run."""

    objective_name: str
    problem_id: str
    task_name: str
    num_traces: int
    exploration_dataset_path: str
    landscape_data_path: str
    baseline_cost: float


class ProblemLandscapeSummary(BaseModel):
    """Root artifact for per-problem landscape runs."""

    items: list[ProblemLandscapeSummaryItem]


class PlacementLandscapePerProblemCommand(BaseCommand):
    """Generate stacked optimal-placement and score comparison plots."""

    config: "PlacementLandscapePerProblemCommandConfig"

    async def run(self) -> None:
        """Execute the per-problem comparison pipeline."""
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)
        explore_dir = Path(self.config.landscape.explore_output_dir)
        manifest_path = explore_dir / "per_problem_manifest.json"

        logger.info("Output directory: %s", output_dir)
        logger.info("Loading per-problem manifest from %s", manifest_path)

        if not manifest_path.exists():
            raise FileNotFoundError(
                "placement_landscape_per_problem requires "
                f"{manifest_path} to exist"
            )

        manifest = PerProblemTraceManifest.model_validate_json(
            manifest_path.read_text()
        )
        if not manifest.items:
            raise ValueError("Per-problem manifest is empty")

        selected_items = manifest.items[: self.config.top_k]
        logger.info(
            "Comparing optimal placements for %d problems (top_k=%d)",
            len(selected_items),
            self.config.top_k,
        )

        summary_items: list[ProblemLandscapeSummaryItem] = []
        landscape_items: list[tuple[ProblemLandscapeSummaryItem, LandscapeData]] = []
        for item in selected_items:
            dataset_path = find_objective_result(
                explore_dir, "exploration_results", item.objective_name
            )
            landscape_data = self._compute_landscape_data(
                obj_name=item.objective_name,
                dataset_path=dataset_path,
            )
            landscape_data_path = output_dir / item.objective_name / "landscape_data.json"
            landscape_data.save(landscape_data_path)
            summary_item = ProblemLandscapeSummaryItem(
                objective_name=item.objective_name,
                problem_id=item.problem_id,
                task_name=item.task_name,
                num_traces=item.num_traces,
                exploration_dataset_path=str(dataset_path),
                landscape_data_path=str(landscape_data_path),
                baseline_cost=landscape_data.baseline_cost,
            )
            summary_items.append(summary_item)
            landscape_items.append((summary_item, landscape_data))

        summary = ProblemLandscapeSummary(items=summary_items)
        (output_dir / "selected_problems.json").write_text(
            summary.model_dump_json(indent=2)
        )

        if landscape_items:
            self._plot_stacked_placements(landscape_items, output_dir)
            self._plot_score_curves(landscape_items, output_dir)

    def _compute_landscape_data(
        self,
        *,
        obj_name: str,
        dataset_path: Path,
    ) -> LandscapeData:
        """Fit a surrogate and compute only the DP-optimal line per cost band."""
        from place_layers.surrogate.dataset import ExplorationDataset

        logger.info("=== Computing optimal-placement bands for %s ===", obj_name)
        dataset = ExplorationDataset.load(dataset_path)
        logger.info(
            "Loaded dataset: %d samples, %d layers, quality=%.4f",
            dataset.num_samples,
            dataset.num_layers,
            dataset.final_quality,
        )

        landscape = self.config.landscape
        cost_model = landscape.cost_model.create_cost_model()

        pooled = PooledEvaluationData.from_exploration(dataset)
        supplemental_files = (
            discover_evaluation_records(
                landscape.supplemental_data_dirs,
                dataset.objective_name,
                num_layers=dataset.num_layers,
                mixer_names=list(dataset.mixer_names),
            )
            if landscape.supplemental_data_dirs
            else []
        )
        if supplemental_files:
            pooled = pooled.merge(PooledEvaluationData.from_jsonl(supplemental_files))

        all_assignments = list(pooled.assignments)
        all_scores = np.array(list(pooled.scores))
        surrogate = landscape.surrogate.create_surrogate(
            mixer_names=list(dataset.mixer_names),
            seed=landscape.seed,
            feature_config=dataset.feature_config,
        )
        match landscape.surrogate:
            case BayesianRidgeSurrogateConfig(alpha_selection="evidence"):
                selected_alpha = surrogate.fit_optimize_alpha(
                    all_assignments, all_scores
                )
                logger.info("Type-II ML alpha for %s: %.4g", obj_name, selected_alpha)
            case _:
                surrogate.fit(all_assignments, all_scores)

        num_layers = dataset.num_layers
        all_allocations = self._enumerate_allocations(cost_model, num_layers)
        alloc_costs = np.array(
            [cost_model.allocation_cost(a) for a in all_allocations]
        )
        baseline_cost = cost_model.max_cost(num_layers)
        logger.info(
            "Enumerated %d allocations for %s (cost range %.4f - %.4f)",
            len(all_allocations),
            obj_name,
            float(alloc_costs.min()),
            float(alloc_costs.max()),
        )

        band_results, cost_bin_edges = compute_dp_per_band(
            surrogate,
            all_allocations,
            alloc_costs,
            num_layers,
            baseline_cost,
            cost_model,
            landscape.num_cost_bins,
        )
        logger.info(
            "Computed %d DP bands for %s",
            len(band_results),
            obj_name,
        )

        return LandscapeData(
            objective_name=obj_name,
            num_layers=num_layers,
            baseline_cost=baseline_cost,
            num_cost_bins=landscape.num_cost_bins,
            cost_bin_edges=cost_bin_edges.tolist(),
            bands=band_results,
        )

    @staticmethod
    def _enumerate_allocations(cost_model, num_layers: int):
        from place_layers.cost_model.enumeration import enumerate_feasible_allocations

        return enumerate_feasible_allocations(
            cost_model, num_layers, cost_budget=None
        )

    def _plot_stacked_placements(
        self,
        items: list[tuple[ProblemLandscapeSummaryItem, LandscapeData]],
        output_dir: Path,
    ) -> None:
        """Render a stacked layermap comparison across selected problems."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
        from matplotlib.patches import Patch

        first_landscape = items[0][1]
        num_problems = len(items)
        num_layers = first_landscape.num_layers
        cost_model = self.config.landscape.cost_model.create_cost_model()
        phase_order = cost_model.phase_order

        mixer_colors = self._resolve_mixer_colors(phase_order)
        mixer_to_idx = {name: i for i, name in enumerate(phase_order)}
        num_mixers = len(phase_order)

        fig_height = max(2.5 + 1.3 * num_problems, 4.5)
        fig, axes = plt.subplots(
            num_problems,
            1,
            figsize=(10, fig_height),
            dpi=300,
            sharex=True,
        )
        if num_problems == 1:
            axes = [axes]

        for ax, (summary_item, landscape) in zip(axes, items, strict=True):
            speedup_bin_edges = np.linspace(
                1.0,
                max(
                    landscape.baseline_cost / band.cost
                    for band in landscape.bands
                ),
                landscape.num_cost_bins + 1,
            )
            layermap_data = self._build_layermap_data(
                landscape.bands,
                landscape.baseline_cost,
                speedup_bin_edges,
                num_layers,
                landscape.num_cost_bins,
                mixer_to_idx,
            )

            cmap_colors = ["white"] + [mixer_colors[name] for name in phase_order]
            mixer_cmap = ListedColormap(cmap_colors)
            boundaries = np.arange(-1.5, num_mixers + 0.5, 1)
            norm = BoundaryNorm(boundaries, len(cmap_colors))
            ax.pcolormesh(
                speedup_bin_edges,
                np.arange(num_layers + 1),
                layermap_data,
                cmap=mixer_cmap,
                norm=norm,
            )
            ax.invert_yaxis()
            ax.set_ylabel(summary_item.problem_id, rotation=0, labelpad=34, va="center")

        axes[-1].set_xlabel("Speedup (x)")
        for ax in axes:
            ax.set_ylim(num_layers, 0)
        fig.suptitle(
            f"Optimal Placements",
            y=0.995,
        )

        legend_patches = [
            Patch(facecolor=mixer_colors[name], label=name)
            for name in phase_order
        ]
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=min(len(phase_order), 5),
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))

        pdf_path = output_dir / "placement_comparison_stacked.pdf"
        png_path = output_dir / "placement_comparison_stacked.png"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s and %s", pdf_path, png_path)

    def _plot_score_curves(
        self,
        items: list[tuple[ProblemLandscapeSummaryItem, LandscapeData]],
        output_dir: Path,
    ) -> None:
        """Render score versus speedup for each selected problem."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        cmap = plt.get_cmap("tab10")

        for index, (summary_item, landscape) in enumerate(items):
            speedups = [landscape.baseline_cost / band.cost for band in landscape.bands]
            scores = [band.score for band in landscape.bands]
            ax.plot(
                speedups,
                scores,
                color=cmap(index % cmap.N),
                lw=2,
                alpha=0.95,
                label=summary_item.problem_id,
            )

        ax.set_xlabel("Speedup (x)")
        ax.set_ylabel(self.config.score_label)
        ax.set_title(
            f"Optimal Score vs Speedup"
        )
        ax.grid(True, alpha=0.2)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
        fig.tight_layout()

        pdf_path = output_dir / "score_comparison.pdf"
        png_path = output_dir / "score_comparison.png"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s and %s", pdf_path, png_path)

    @staticmethod
    def _resolve_mixer_colors(phase_order: list[str]) -> dict[str, str]:
        canonical = {
            "attention": "tab:blue",
            "sliding_window": "tab:orange",
            "kda": "tab:green",
            "gdn": "tab:red",
            "mamba": "tab:purple",
        }
        fallback = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
        ]
        colors: dict[str, str] = {}
        fallback_idx = 0
        for name in phase_order:
            if name in canonical:
                colors[name] = canonical[name]
            else:
                colors[name] = fallback[fallback_idx % len(fallback)]
                fallback_idx += 1
        return colors

    @staticmethod
    def _build_layermap_data(
        layermap_source: list[BandResult],
        baseline_cost: float,
        speedup_bin_edges: np.ndarray,
        num_layers: int,
        num_bins: int,
        mixer_to_idx: dict[str, int],
    ) -> np.ndarray:
        """Build (num_layers, num_bins) array using range-based bin filling."""
        layermap_data = np.full((num_layers, num_bins), -1, dtype=int)
        for br in layermap_source:
            sp_lo = baseline_cost / br.cost_high
            sp_hi = baseline_cost / br.cost_low
            for b in range(num_bins):
                if speedup_bin_edges[b] < sp_hi and speedup_bin_edges[b + 1] > sp_lo:
                    for layer_idx, mixer_name in enumerate(br.assignments):
                        layermap_data[layer_idx, b] = mixer_to_idx[mixer_name]
        return layermap_data


class PlacementLandscapePerProblemCommandConfig(BaseCommandConfig):
    """Configuration for placement_landscape_per_problem command."""

    _impl_class: ClassVar[type[BaseCommand]] = PlacementLandscapePerProblemCommand

    command: Literal["placement_landscape_per_problem"] = (
        "placement_landscape_per_problem"
    )

    landscape: LandscapeConfig = Field(
        description="Shared landscape configuration (binning, surrogate, cost model).",
    )

    score_label: str = Field(
        description="Y-axis label for the score comparison plot.",
    )

    top_k: int = Field(
        default=10,
        description="Number of problems from manifest order to compare.",
        gt=0,
    )

    samples_per_allocation: int = Field(
        default=1,
        description="Unused legacy field kept for config compatibility.",
        gt=0,
    )

    num_score_bins: int = Field(
        default=1,
        description="Unused legacy field kept for config compatibility.",
        gt=0,
    )

    display_names: dict[str, str] | None = Field(
        default=None,
        description="Unused legacy field kept for config compatibility.",
    )

    validation_output_dir: str | None = Field(
        default=None,
        description="Unused legacy field kept for config compatibility.",
    )
