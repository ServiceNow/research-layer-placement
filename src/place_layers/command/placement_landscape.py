"""Placement landscape command: density plot of cost-vs-quality landscape.

Generates a paper-quality figure showing:
1. Density of random placements — most configurations at any cost are bad
2. DP-optimal line — the surrogate's Viterbi solution at each cost level
3. (Optional) Validated line — real evaluated scores from validate_landscape

Input: exploration_results.json from explore_placements + a cost model config.
       Optionally: validation_results.json from validate_landscape.
Output: placement_landscape.pdf + .png + landscape_data.json in Hydra output dir.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from hydra.core.hydra_config import HydraConfig
from pydantic import Field

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.command.landscape_config import LandscapeConfig
from place_layers.command.landscape_helpers import (
    BandResult,
    LandscapeData,
    ValidationResults,
    compute_band_edges,
    compute_dp_per_band,
    load_and_fit_surrogate,
)
from place_layers.cost_model.base_cost_model import Allocation, BaseCostModel
from place_layers.cost_model.enumeration import enumerate_feasible_allocations

logger = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================


def _sample_permutations(
    rng: np.random.Generator,
    allocation: dict[str, int],
    num_layers: int,
    n: int,
) -> list[tuple[str, ...]]:
    """Generate n random permutations of an allocation using numpy.

    Args:
        rng: Numpy random generator.
        allocation: Mixer counts (e.g. {"attention": 10, "gdn": 38}).
        num_layers: Total layers (must equal sum of allocation values).
        n: Number of permutations to generate.

    Returns:
        List of n assignment tuples.
    """
    base = np.array(
        [m for m, c in sorted(allocation.items()) for _ in range(c)]
    )
    return [tuple(base[rng.permutation(num_layers)]) for _ in range(n)]


# ============================================================================
# Command
# ============================================================================


class PlacementLandscapeCommand(BaseCommand):
    """Generate cost-vs-quality density plot with DP-optimal line."""

    config: "PlacementLandscapeCommandConfig"

    async def run(self) -> None:
        """Execute placement landscape generation."""
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)
        logger.info("Output directory: %s", output_dir)

        landscape = self.config.landscape
        dataset, surrogate, cost_model = load_and_fit_surrogate(
            landscape.exploration_dataset_path,
            landscape.cost_model,
            landscape.seed,
        )
        num_layers = dataset.num_layers

        # Enumerate ALL allocations (no budget filter)
        all_allocations = enumerate_feasible_allocations(
            cost_model, num_layers, cost_budget=None
        )
        logger.info(
            "Enumerated %d allocations for %d layers",
            len(all_allocations),
            num_layers,
        )

        # Compute cost range and baseline for speedup conversion
        alloc_costs = np.array(
            [cost_model.allocation_cost(a) for a in all_allocations]
        )
        baseline_cost = cost_model.max_cost(num_layers)
        logger.info(
            "Cost range: %.4f – %.4f (baseline=%.4f)",
            float(alloc_costs.min()),
            float(alloc_costs.max()),
            baseline_cost,
        )

        # Generate density samples
        costs, scores = self._generate_density_samples(
            surrogate, all_allocations, alloc_costs, num_layers
        )
        logger.info(
            "Generated %d density samples (score range: %.4f – %.4f)",
            len(costs),
            float(scores.min()),
            float(scores.max()),
        )

        # Compute DP-optimal per cost band
        band_results = compute_dp_per_band(
            surrogate, all_allocations, alloc_costs,
            num_layers, baseline_cost, cost_model,
            landscape.num_cost_bins,
        )
        logger.info(
            "DP solutions for %d / %d cost bands",
            len(band_results),
            landscape.num_cost_bins,
        )
        if band_results:
            logger.info(
                "Pareto line: cost %.4f (score %.4f) -> cost %.4f (score %.4f)",
                band_results[0].cost,
                band_results[0].score,
                band_results[-1].cost,
                band_results[-1].score,
            )

        # Save band data as JSON
        cost_bin_edges = compute_band_edges(
            alloc_costs, baseline_cost, landscape.num_cost_bins,
        )
        landscape_data = LandscapeData(
            num_layers=num_layers,
            baseline_cost=baseline_cost,
            num_cost_bins=landscape.num_cost_bins,
            cost_bin_edges=cost_bin_edges.tolist(),
            bands=band_results,
        )
        landscape_data.save(output_dir / "landscape_data.json")
        logger.info("Saved landscape_data.json")

        # Load validation data if provided
        validation: ValidationResults | None = None
        if self.config.validation_results_path is not None:
            validation = ValidationResults.load(
                Path(self.config.validation_results_path)
            )
            logger.info(
                "Loaded validation results: %d bands, rho=%.4f",
                len(validation.bands),
                validation.summary.spearman_rho,
            )

        # Observed data overlay
        observed_costs: np.ndarray | None = None
        observed_scores: np.ndarray | None = None
        if self.config.show_observed:
            observed_costs = np.array(
                [cost_model.allocation_cost(a) for a in dataset.allocations]
            )
            observed_scores = np.array(dataset.scores)

        # Plot
        self._plot(
            costs,
            scores,
            band_results,
            baseline_cost,
            cost_model,
            num_layers,
            observed_costs,
            observed_scores,
            output_dir,
            validation,
        )

    def _generate_density_samples(
        self,
        surrogate,
        all_allocations: list[Allocation],
        alloc_costs: np.ndarray,
        num_layers: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample random permutations for all allocations and predict scores.

        Returns:
            (costs, scores) arrays of shape (total_samples,).
        """
        rng = np.random.default_rng(self.config.landscape.seed)
        n_per = self.config.samples_per_allocation
        chunk_size = 10_000

        all_costs: list[float] = []
        all_scores_list: list[np.ndarray] = []

        # Buffer for chunked prediction
        chunk_assignments: list[tuple[str, ...]] = []
        chunk_costs: list[float] = []

        for i, alloc in enumerate(all_allocations):
            cost = float(alloc_costs[i])
            perms = _sample_permutations(rng, alloc, num_layers, n_per)
            chunk_assignments.extend(perms)
            chunk_costs.extend([cost] * n_per)

            # Flush when chunk is full
            if len(chunk_assignments) >= chunk_size:
                preds = surrogate.predict_mean(chunk_assignments)
                all_scores_list.append(preds)
                all_costs.extend(chunk_costs)
                chunk_assignments = []
                chunk_costs = []

        # Flush remaining
        if chunk_assignments:
            preds = surrogate.predict_mean(chunk_assignments)
            all_scores_list.append(preds)
            all_costs.extend(chunk_costs)

        return np.array(all_costs), np.concatenate(all_scores_list)

    def _plot(
        self,
        costs: np.ndarray,
        scores: np.ndarray,
        band_results: list[BandResult],
        baseline_cost: float,
        cost_model: BaseCostModel,
        num_layers: int,
        observed_costs: np.ndarray | None,
        observed_scores: np.ndarray | None,
        output_dir: Path,
        validation: ValidationResults | None = None,
    ) -> None:
        """Render two-panel figure: density + layermap."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from matplotlib.patches import Patch

        cfg = self.config
        phase_order = cost_model.phase_order

        # Convert costs to speedup (presentation only)
        speedups = baseline_cost / costs
        min_speedup = 1.0
        max_speedup = float(speedups.max())

        # Mixer color map: one color per mixer type by phase_order index
        tab_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
                       "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
        mixer_colors = {
            name: tab_colors[i % len(tab_colors)]
            for i, name in enumerate(phase_order)
        }

        # GridSpec layout: density plot + colorbar column, layermap below
        fig = plt.figure(figsize=(7, 7), dpi=300)
        gs = fig.add_gridspec(
            2, 2, height_ratios=[3, 1], width_ratios=[1, 0.03],
            hspace=0.05, wspace=0.05,
        )
        ax_density = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        ax_layermap = fig.add_subplot(gs[1, 0], sharex=ax_density)

        # ---- Top panel: density + DP line ----
        h = ax_density.hist2d(
            speedups, scores,
            bins=[cfg.landscape.num_cost_bins, cfg.num_score_bins],
            cmap="Greys",
            norm=LogNorm(),
            density=True,
            range=[[min_speedup, max_speedup], [0.0, 1.0]],
        )
        fig.colorbar(h[3], cax=cax, label="Density")
        ax_density.set_ylim(0.0, 1.0)

        # DP-optimal line (dashed when validation present, solid otherwise)
        if band_results:
            band_speedups = [baseline_cost / r.cost for r in band_results]
            band_scores = [r.score for r in band_results]
            dp_style = "--" if validation is not None else "-"
            dp_label = "Surrogate DP" if validation is not None else "DP optimum"
            ax_density.plot(
                band_speedups, band_scores,
                color="tab:blue", lw=2, ls=dp_style,
                label=dp_label, zorder=5,
            )

        # Validated line (when validation data present)
        if validation is not None and validation.bands:
            val_speedups = [
                baseline_cost / b.validated.cost for b in validation.bands
            ]
            val_scores = [b.validated.score for b in validation.bands]
            ax_density.plot(
                val_speedups, val_scores,
                color="tab:red", lw=2, ls="-",
                label="Validated", zorder=6,
            )

        # Observed data overlay
        if observed_costs is not None and observed_scores is not None:
            observed_speedups = baseline_cost / observed_costs
            ax_density.scatter(
                observed_speedups, observed_scores,
                s=3, alpha=0.3, color="tab:red" if validation is None else "tab:gray",
                zorder=3,
                label="Observed",
            )

        # Random chance line
        ax_density.axhline(
            y=0.25, color="tab:blue", ls="--", lw=1.5, zorder=2,
        )
        ax_density.text(
            min_speedup + 0.02 * (max_speedup - min_speedup), 0.25 + 0.005,
            "Random chance", fontsize=8, color="tab:blue", va="bottom",
        )

        ax_density.set_ylabel(cfg.score_label)
        ax_density.legend(loc="upper left", fontsize=8)
        ax_density.tick_params(labelbottom=False)

        # ---- Bottom panel: layermap ----
        # Use validated assignments if available, otherwise surrogate DP
        if validation is not None and validation.bands:
            layermap_source = [
                BandResult(
                    band_index=b.band_index,
                    cost_low=b.cost_low,
                    cost_high=b.cost_high,
                    cost=b.validated.cost,
                    score=b.validated.score,
                    assignments=b.validated.assignments,
                )
                for b in validation.bands
            ]
        else:
            layermap_source = band_results

        num_bins = cfg.landscape.num_cost_bins
        speedup_bin_edges = np.linspace(min_speedup, max_speedup, num_bins + 1)
        layer_edges = np.arange(num_layers + 1)

        # Build mixer name -> index mapping from phase_order
        mixer_to_idx = {name: i for i, name in enumerate(phase_order)}
        num_mixers = len(phase_order)

        # Build layermap array: (num_layers, num_bins), fill with -1 (empty)
        layermap_data = np.full((num_layers, num_bins), -1, dtype=int)
        for br in layermap_source:
            br_speedup = baseline_cost / br.cost
            b = int(np.digitize(br_speedup, speedup_bin_edges)) - 1
            b = min(b, num_bins - 1)
            for layer_idx, mixer_name in enumerate(br.assignments):
                layermap_data[layer_idx, b] = mixer_to_idx[mixer_name]

        # Colormap: white for empty (-1), then one color per mixer in phase_order
        from matplotlib.colors import BoundaryNorm, ListedColormap
        cmap_colors = ["white"] + [mixer_colors[name] for name in phase_order]
        mixer_cmap = ListedColormap(cmap_colors)
        boundaries = np.arange(-1.5, num_mixers + 0.5, 1)
        norm = BoundaryNorm(boundaries, len(cmap_colors))

        ax_layermap.pcolormesh(
            speedup_bin_edges, layer_edges, layermap_data,
            cmap=mixer_cmap, norm=norm,
        )
        ax_layermap.set_ylabel("Layer index")
        ax_layermap.set_xlabel("Speedup (\u00d7)")
        ax_layermap.invert_yaxis()

        # Legend for mixer types
        legend_patches = [
            Patch(facecolor=mixer_colors[name], label=name)
            for name in phase_order
        ]
        ax_layermap.legend(
            handles=legend_patches,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.4),
            ncol=min(len(phase_order), 4),
            fontsize=8,
        )

        # Save
        pdf_path = output_dir / "placement_landscape.pdf"
        png_path = output_dir / "placement_landscape.png"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s and %s", pdf_path, png_path)


# ============================================================================
# Config
# ============================================================================


class PlacementLandscapeCommandConfig(BaseCommandConfig):
    """Configuration for placement_landscape command."""

    _impl_class: ClassVar[type[BaseCommand]] = PlacementLandscapeCommand

    command: Literal["placement_landscape"] = "placement_landscape"

    landscape: LandscapeConfig = Field(
        description="Shared landscape configuration (binning, surrogate, cost model).",
    )

    # Sampling
    samples_per_allocation: int = Field(
        description="Random permutations per allocation for density estimation.",
        gt=0,
    )

    # Plot
    num_score_bins: int = Field(
        description="Histogram bins along score axis.",
        gt=0,
    )
    show_observed: bool = Field(
        description="Overlay real evaluation points from exploration dataset.",
    )
    score_label: str = Field(
        description="Y-axis label for the density plot.",
    )

    # Validation overlay (optional — YAML supplies null)
    validation_results_path: str | None = Field(
        description="Path to validation_results.json from validate_landscape. "
        "If provided, overlays real evaluated scores and uses validated placements in layermap.",
    )
