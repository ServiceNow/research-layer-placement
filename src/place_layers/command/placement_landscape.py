"""Placement landscape command: density plot of cost-vs-quality landscape.

Generates a paper-quality figure per objective showing:
1. Density of random placements — most configurations at any cost are bad
2. DP-optimal line — the surrogate's Viterbi solution at each cost level
3. (Optional) Validated line — real evaluated scores from validate_landscape

Input: explore_output_dir (contains exploration_results_{obj_name}.json per objective)
       + evaluators config (for objective_names) + a cost model config.
       Optionally: validation_output_dir with validation_results_{obj_name}.json.
Output: placement_landscape_{obj_name}.pdf/png + landscape_data_{obj_name}.json per objective.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar, Literal, assert_never

import numpy as np
from hydra.core.hydra_config import HydraConfig
from pydantic import Field

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.landscape import (
    BandResult,
    LandscapeConfig,
    LandscapeData,
    ValidationResults,
    compute_dp_per_band,
    objective_result_path,
)
from place_layers.surrogate.evaluation_log import (
    discover_evaluation_records,
    load_pooled_data,
)
from place_layers.cost_model.base_cost_model import Allocation, BaseCostModel
from place_layers.cost_model.enumeration import enumerate_feasible_allocations
from place_layers.evaluator.evaluators import EvaluatorsConfig
from place_layers.surrogate.bayesian_ridge import BayesianRidgeSurrogate
from place_layers.surrogate.config import (
    BayesianRidgeSurrogateConfig,
    RidgeSurrogateConfig,
)
from place_layers.surrogate.ridge_surrogate import RidgeSurrogate

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
        """Execute placement landscape generation for each objective."""
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)
        logger.info("Output directory: %s", output_dir)

        landscape = self.config.landscape
        explore_dir = Path(landscape.explore_output_dir)

        for obj_name in self.config.evaluators.objective_names:
            dataset_path = objective_result_path(
                explore_dir, "exploration_results", obj_name,
            )
            await self._generate_landscape(
                obj_name=obj_name,
                dataset_path=dataset_path,
                output_dir=output_dir,
            )

    async def _generate_landscape(
        self,
        *,
        obj_name: str,
        dataset_path: Path,
        output_dir: Path,
    ) -> None:
        """Generate landscape for a single objective."""
        logger.info("=== Generating landscape for objective: %s ===", obj_name)

        landscape = self.config.landscape

        # Auto-include validation_output_dir in supplemental dirs for pooling
        supplemental_dirs = list(landscape.supplemental_data_dirs)
        if self.config.validation_output_dir is not None:
            val_dir = self.config.validation_output_dir
            if val_dir not in supplemental_dirs:
                supplemental_dirs.append(val_dir)

        from place_layers.surrogate.dataset import ExplorationDataset

        dataset = ExplorationDataset.load(dataset_path)
        logger.info(
            "Loaded dataset: %d samples, %d layers, rho=%.4f",
            dataset.num_samples,
            dataset.num_layers,
            dataset.final_rho,
        )
        cost_model = landscape.cost_model.create_cost_model()

        supplemental_files = (
            discover_evaluation_records(supplemental_dirs, dataset.objective_name)
            if supplemental_dirs else []
        )
        pooled = load_pooled_data(
            [dataset_path, *supplemental_files], deduplicate=True,
        )
        surrogate = landscape.surrogate.create_surrogate(
            mixer_names=pooled.mixer_names, seed=landscape.seed,
            feature_config=dataset.feature_config,
        )
        surrogate.fit(pooled.assignments, np.array(pooled.scores))
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
        band_results, cost_bin_edges = compute_dp_per_band(
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
        landscape_data = LandscapeData(
            objective_name=obj_name,
            num_layers=num_layers,
            baseline_cost=baseline_cost,
            num_cost_bins=landscape.num_cost_bins,
            cost_bin_edges=cost_bin_edges.tolist(),
            bands=band_results,
        )
        landscape_data.save(
            objective_result_path(output_dir, "landscape_data", obj_name)
        )
        logger.info("Saved %s/landscape_data.json", obj_name)

        # Load validation data if provided
        validation: ValidationResults | None = None
        if self.config.validation_output_dir is not None:
            validation_path = objective_result_path(
                Path(self.config.validation_output_dir),
                "validation_results",
                obj_name,
            )
            if validation_path.exists():
                validation = ValidationResults.load(validation_path)
                logger.info(
                    "Loaded validation results for %s: %d bands",
                    obj_name,
                    len(validation.bands),
                )
                # Filter out bands with validated cost outside their band range
                in_band = [
                    b for b in validation.bands
                    if b.cost_low <= b.validated.cost <= b.cost_high
                ]
                n_dropped = len(validation.bands) - len(in_band)
                if n_dropped > 0:
                    logger.warning(
                        "Filtered %d / %d validation bands with out-of-range "
                        "validated costs for %s",
                        n_dropped,
                        len(validation.bands),
                        obj_name,
                    )
                    validation = ValidationResults(
                        objective_name=validation.objective_name,
                        num_layers=validation.num_layers,
                        baseline_cost=validation.baseline_cost,
                        num_cost_bins=validation.num_cost_bins,
                        cost_bin_edges=validation.cost_bin_edges,
                        bands=in_band,
                        summary=validation.summary,
                    )
            else:
                logger.warning(
                    "Validation results not found for %s at %s, skipping overlay",
                    obj_name,
                    validation_path,
                )

        # Observed data overlay (always shown) — pooled already has exploration + supplemental
        observed_costs = np.array(
            [cost_model.allocation_cost(a) for a in pooled.allocations]
        )
        observed_scores = np.array(pooled.scores)

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
            obj_name,
            validation,
            surrogate,
        )

    def _generate_density_samples(
        self,
        surrogate: RidgeSurrogate | BayesianRidgeSurrogate,
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
        obj_name: str,
        validation: ValidationResults | None,
        surrogate: RidgeSurrogate | BayesianRidgeSurrogate,
    ) -> None:
        """Render two-panel figure: density + layermap."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cfg = self.config
        phase_order = cost_model.phase_order

        # Resolve Y-axis label
        if cfg.display_names and obj_name in cfg.display_names:
            score_label = cfg.display_names[obj_name]
        else:
            score_label = f"{cfg.score_label} ({obj_name})"

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

        # Top panel: density + DP line + uncertainty + validation + observed
        self._plot_density_panel(
            ax=ax_density,
            fig=fig,
            cax=cax,
            speedups=speedups,
            scores=scores,
            band_results=band_results,
            baseline_cost=baseline_cost,
            min_speedup=min_speedup,
            max_speedup=max_speedup,
            observed_costs=observed_costs,
            observed_scores=observed_scores,
            score_label=score_label,
            validation=validation,
            surrogate=surrogate,
        )

        # Bottom panel: layermap
        self._plot_layermap_panel(
            ax=ax_layermap,
            band_results=band_results,
            baseline_cost=baseline_cost,
            min_speedup=min_speedup,
            max_speedup=max_speedup,
            num_layers=num_layers,
            phase_order=phase_order,
            mixer_colors=mixer_colors,
            validation=validation,
        )

        # Save
        obj_dir = output_dir / obj_name
        obj_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = obj_dir / "placement_landscape.pdf"
        png_path = obj_dir / "placement_landscape.png"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s and %s", pdf_path, png_path)

    def _plot_density_panel(
        self,
        *,
        ax,
        fig,
        cax,
        speedups: np.ndarray,
        scores: np.ndarray,
        band_results: list[BandResult],
        baseline_cost: float,
        min_speedup: float,
        max_speedup: float,
        observed_costs: np.ndarray | None,
        observed_scores: np.ndarray | None,
        score_label: str,
        validation: ValidationResults | None,
        surrogate: RidgeSurrogate | BayesianRidgeSurrogate,
    ) -> None:
        """Render density histogram, DP line, uncertainty band, validation, observed."""
        from matplotlib.colors import LogNorm

        cfg = self.config

        # Compute dynamic y-axis range from all score sources
        all_score_vals: list[np.ndarray] = [scores]
        if band_results:
            band_scores_arr = np.array([r.score for r in band_results])
            all_score_vals.append(band_scores_arr)
            # Include ±2σ uncertainty band extremes so fill_between isn't clipped
            _, band_sigma = surrogate.predict(
                [br.assignments for br in band_results]
            )
            all_score_vals.append(band_scores_arr - 2 * band_sigma)
            all_score_vals.append(band_scores_arr + 2 * band_sigma)
        if validation is not None and validation.bands:
            all_score_vals.append(
                np.array([b.validated.score for b in validation.bands])
            )
        if observed_scores is not None and len(observed_scores) > 0:
            all_score_vals.append(observed_scores)

        combined = np.concatenate(all_score_vals)
        score_min = float(combined.min())
        score_max = float(combined.max())
        margin = 0.05 * (score_max - score_min) if score_max > score_min else 0.5
        score_lo = score_min - margin
        score_hi = score_max + margin

        h = ax.hist2d(
            speedups, scores,
            bins=[cfg.landscape.num_cost_bins, cfg.num_score_bins],
            cmap="Greys",
            norm=LogNorm(),
            density=True,
            range=[[min_speedup, max_speedup], [score_lo, score_hi]],
        )
        fig.colorbar(h[3], cax=cax, label="Density")
        ax.set_ylim(score_lo, score_hi)

        # DP-optimal line (dashed when validation present, solid otherwise)
        if band_results:
            band_speedups = [baseline_cost / r.cost for r in band_results]
            band_scores = [r.score for r in band_results]
            dp_style = "--" if validation is not None else "-"
            dp_label = "Surrogate DP" if validation is not None else "DP optimum"
            ax.plot(
                band_speedups, band_scores,
                color="tab:blue", lw=2, ls=dp_style,
                label=dp_label, zorder=5,
            )

            # Uncertainty band
            self._plot_uncertainty_band(
                ax=ax,
                band_results=band_results,
                baseline_cost=baseline_cost,
                surrogate=surrogate,
            )

        # Validated line (when validation data present)
        if validation is not None and validation.bands:
            val_speedups = [
                baseline_cost / b.validated.cost for b in validation.bands
            ]
            val_scores = [b.validated.score for b in validation.bands]
            ax.plot(
                val_speedups, val_scores,
                color="tab:red", lw=2, ls="-",
                label="Validated", zorder=6,
            )

        # Observed data overlay
        if observed_costs is not None and observed_scores is not None:
            observed_speedups = baseline_cost / observed_costs
            ax.scatter(
                observed_speedups, observed_scores,
                s=3, alpha=0.3, color="tab:red" if validation is None else "tab:gray",
                zorder=3,
                label="Observed",
            )

        ax.set_ylabel(score_label)
        ax.legend(loc="upper left", fontsize=8)
        ax.tick_params(labelbottom=False)

    def _plot_uncertainty_band(
        self,
        *,
        ax,
        band_results: list[BandResult],
        baseline_cost: float,
        surrogate: RidgeSurrogate | BayesianRidgeSurrogate,
    ) -> None:
        """Plot surrogate uncertainty as fill_between around DP line."""
        assignments = [br.assignments for br in band_results]
        _, sigma = surrogate.predict(assignments)
        # Center on DP scores (not predict mean) so band matches the plotted line
        band_scores = np.array([r.score for r in band_results])
        band_speedups = np.array([baseline_cost / r.cost for r in band_results])

        match self.config.landscape.surrogate:
            case BayesianRidgeSurrogateConfig():
                label = "Posterior \u00b12\u03c3"
            case RidgeSurrogateConfig():
                label = "Bootstrap \u00b12\u03c3"
            case other:
                assert_never(other)

        ax.fill_between(
            band_speedups,
            band_scores - 2 * sigma,
            band_scores + 2 * sigma,
            alpha=0.15, color="tab:blue", label=label, zorder=4,
        )

    def _plot_layermap_panel(
        self,
        *,
        ax,
        band_results: list[BandResult],
        baseline_cost: float,
        min_speedup: float,
        max_speedup: float,
        num_layers: int,
        phase_order: list[str],
        mixer_colors: dict[str, str],
        validation: ValidationResults | None,
    ) -> None:
        """Render layermap pcolormesh with mixer legend."""
        from matplotlib.colors import BoundaryNorm, ListedColormap
        from matplotlib.patches import Patch

        cfg = self.config
        num_bins = cfg.landscape.num_cost_bins
        speedup_bin_edges = np.linspace(min_speedup, max_speedup, num_bins + 1)
        layer_edges = np.arange(num_layers + 1)

        mixer_to_idx = {name: i for i, name in enumerate(phase_order)}
        num_mixers = len(phase_order)

        # Merge validation + DP fallback for layermap source
        if validation is not None and validation.bands:
            validated_by_index = {b.band_index: b for b in validation.bands}
            merged: list[BandResult] = []
            for br in band_results:
                if br.band_index in validated_by_index:
                    vb = validated_by_index[br.band_index]
                    merged.append(BandResult(
                        band_index=vb.band_index,
                        cost_low=vb.cost_low,
                        cost_high=vb.cost_high,
                        cost=vb.validated.cost,
                        score=vb.validated.score,
                        assignments=vb.validated.assignments,
                    ))
                else:
                    merged.append(br)  # DP fallback
            layermap_source = merged
        else:
            layermap_source = band_results

        layermap_data = self._build_layermap_data(
            layermap_source, baseline_cost, speedup_bin_edges,
            num_layers, num_bins, mixer_to_idx,
        )

        # Colormap: white for empty (-1), then one color per mixer in phase_order
        cmap_colors = ["white"] + [mixer_colors[name] for name in phase_order]
        mixer_cmap = ListedColormap(cmap_colors)
        boundaries = np.arange(-1.5, num_mixers + 0.5, 1)
        norm = BoundaryNorm(boundaries, len(cmap_colors))

        ax.pcolormesh(
            speedup_bin_edges, layer_edges, layermap_data,
            cmap=mixer_cmap, norm=norm,
        )
        ax.set_ylabel("Layer index")
        ax.set_xlabel("Speedup (\u00d7)")
        ax.invert_yaxis()

        # Legend for mixer types
        legend_patches = [
            Patch(facecolor=mixer_colors[name], label=name)
            for name in phase_order
        ]
        ax.legend(
            handles=legend_patches,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.4),
            ncol=min(len(phase_order), 4),
            fontsize=8,
        )

    @staticmethod
    def _build_layermap_data(
        layermap_source: list[BandResult],
        baseline_cost: float,
        speedup_bin_edges: np.ndarray,
        num_layers: int,
        num_bins: int,
        mixer_to_idx: dict[str, int],
    ) -> np.ndarray:
        """Build (num_layers, num_bins) array using range-based bin filling.

        Each BandResult's [cost_low, cost_high] range is converted to a speedup
        range and ALL overlapping bins are filled, eliminating white gaps.
        """
        layermap_data = np.full((num_layers, num_bins), -1, dtype=int)
        for br in layermap_source:
            sp_lo = baseline_cost / br.cost_high
            sp_hi = baseline_cost / br.cost_low
            for b in range(num_bins):
                if speedup_bin_edges[b] < sp_hi and speedup_bin_edges[b + 1] > sp_lo:
                    for layer_idx, mixer_name in enumerate(br.assignments):
                        layermap_data[layer_idx, b] = mixer_to_idx[mixer_name]
        return layermap_data


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

    evaluators: EvaluatorsConfig = Field(
        description="Evaluator configuration (config-only, used for objective_names).",
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
    score_label: str = Field(
        description="Y-axis label for the density plot.",
    )

    # Display
    display_names: dict[str, str] | None = Field(
        default=None,
        description="Mapping from objective name to human-readable Y-axis label. "
        "When null, falls back to '{score_label} ({obj_name})'.",
    )

    # Validation overlay (optional — YAML supplies null)
    validation_output_dir: str | None = Field(
        default=None,
        description="Path to the output directory from validate_landscape "
        "(contains {obj_name}/validation_results.json files). "
        "If provided, overlays real evaluated scores and uses validated placements in layermap.",
    )

