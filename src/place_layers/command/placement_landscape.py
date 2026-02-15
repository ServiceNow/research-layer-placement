"""Placement landscape command: density plot of cost-vs-quality landscape.

Generates a paper-quality figure showing:
1. Density of random placements — most configurations at any cost are bad
2. DP-optimal line — the surrogate's Viterbi solution at each cost level

Input: exploration_results.json from explore_placements + a cost model config.
Output: placement_landscape.pdf + .png in Hydra output dir.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from hydra.core.hydra_config import HydraConfig
from pydantic import Field

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.cost_model import CostModelConfig
from place_layers.cost_model.base_cost_model import Allocation, BaseCostModel
from place_layers.cost_model.enumeration import enumerate_feasible_allocations
from place_layers.mrf.solver import ExactMRFSolver
from place_layers.surrogate.dataset import ExplorationDataset
from place_layers.surrogate.model_selection import select_expansion_level
from place_layers.surrogate.ridge_surrogate import RidgeSurrogate

logger = logging.getLogger(__name__)


# ============================================================================
# Data structures
# ============================================================================


@dataclass(frozen=True)
class BandResult:
    """DP-optimal result for one cost band."""

    cost: float
    score: float
    assignments: tuple[str, ...]


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

        dataset, surrogate, cost_model = self._load_and_fit_surrogate()
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

        # Compute cost range
        alloc_costs = np.array(
            [cost_model.allocation_cost(a) for a in all_allocations]
        )
        min_cost = float(alloc_costs.min())
        max_cost = float(alloc_costs.max())
        logger.info("Cost range: %.4f – %.4f", min_cost, max_cost)

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
        band_results = self._compute_dp_per_band(
            surrogate, all_allocations, alloc_costs,
            num_layers, min_cost, max_cost,
        )
        logger.info(
            "DP solutions for %d / %d cost bands",
            len(band_results),
            self.config.num_cost_bins,
        )
        if band_results:
            logger.info(
                "Pareto line: cost %.4f (score %.4f) -> cost %.4f (score %.4f)",
                band_results[0].cost,
                band_results[0].score,
                band_results[-1].cost,
                band_results[-1].score,
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
            min_cost,
            max_cost,
            cost_model,
            num_layers,
            observed_costs,
            observed_scores,
            output_dir,
        )

    def _load_and_fit_surrogate(
        self,
    ) -> tuple[ExplorationDataset, RidgeSurrogate, BaseCostModel]:
        """Load dataset, select expansion level, fit surrogate."""
        dataset = ExplorationDataset.load(
            Path(self.config.exploration_dataset_path)
        )
        logger.info(
            "Loaded dataset: %d samples, %d layers, rho=%.4f",
            dataset.num_samples,
            dataset.num_layers,
            dataset.final_rho,
        )

        cost_model = self.config.cost_model.create_cost_model()

        # Model selection
        scores_array = np.array(dataset.scores)
        expansion_result = select_expansion_level(
            assignments=dataset.assignments,
            scores=scores_array,
            mixer_names=dataset.mixer_names,
            seed=self.config.seed,
        )
        feature_config = (
            expansion_result.selected_config
            if expansion_result.selected_config.max_order > 1
            else None
        )
        logger.info(
            "Model selection: order=%d, range=%d, rho=%.4f",
            expansion_result.selected_order,
            expansion_result.selected_range,
            expansion_result.range_results[-1].rho
            if expansion_result.range_results
            else 0.0,
        )

        # Fit surrogate
        surrogate = RidgeSurrogate(
            mixer_names=dataset.mixer_names,
            seed=self.config.seed,
            feature_config=feature_config,
        )
        surrogate.fit(dataset.assignments, scores_array)
        logger.info("Surrogate fitted")

        return dataset, surrogate, cost_model

    def _generate_density_samples(
        self,
        surrogate: RidgeSurrogate,
        all_allocations: list[Allocation],
        alloc_costs: np.ndarray,
        num_layers: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample random permutations for all allocations and predict scores.

        Returns:
            (costs, scores) arrays of shape (total_samples,).
        """
        rng = np.random.default_rng(self.config.seed)
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

    def _compute_dp_per_band(
        self,
        surrogate: RidgeSurrogate,
        all_allocations: list[Allocation],
        alloc_costs: np.ndarray,
        num_layers: int,
        min_cost: float,
        max_cost: float,
    ) -> list[BandResult]:
        """Solve DP per allocation, pick best per cost band.

        Returns:
            List of BandResult sorted by cost, one per occupied band.
        """
        num_bins = self.config.num_cost_bins
        bin_edges = np.linspace(min_cost, max_cost, num_bins + 1)

        # Assign each allocation to a bin
        bin_indices = np.digitize(alloc_costs, bin_edges) - 1
        # Clip: allocations exactly at max_cost go to last bin
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        # Extract MRF model and solver
        mrf_model = surrogate.extract_mrf_model(num_layers)
        solver = ExactMRFSolver(mrf_model)
        intercept = mrf_model.intercept

        # Solve DP for each allocation, track best per bin
        best_per_bin: dict[int, BandResult] = {}

        for i, alloc in enumerate(all_allocations):
            solution = solver.solve_constrained(alloc)
            if solution is None:
                continue

            # DP returns potentials-only score; add intercept
            score = solution.true_score + intercept
            cost = float(alloc_costs[i])
            b = int(bin_indices[i])

            if b not in best_per_bin or score > best_per_bin[b].score:
                best_per_bin[b] = BandResult(
                    cost=cost,
                    score=score,
                    assignments=solution.placement.assignments,
                )

        # Sort by cost
        return [
            best_per_bin[b] for b in sorted(best_per_bin.keys())
        ]

    def _plot(
        self,
        costs: np.ndarray,
        scores: np.ndarray,
        band_results: list[BandResult],
        min_cost: float,
        max_cost: float,
        cost_model: BaseCostModel,
        num_layers: int,
        observed_costs: np.ndarray | None,
        observed_scores: np.ndarray | None,
        output_dir: Path,
    ) -> None:
        """Render two-panel figure: density + layermap."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from matplotlib.patches import Patch

        cfg = self.config
        phase_order = cost_model.phase_order

        # Mixer color map: one color per mixer type by phase_order index
        tab_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
                       "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
        mixer_colors = {
            name: tab_colors[i % len(tab_colors)]
            for i, name in enumerate(phase_order)
        }

        fig, (ax_density, ax_layermap) = plt.subplots(
            2, 1, figsize=(7, 7), dpi=300,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0},
            sharex=True,
        )

        # ---- Top panel: density + DP line ----
        score_min = float(scores.min()) - 0.01
        score_max = float(scores.max()) + 0.01
        h = ax_density.hist2d(
            costs, scores,
            bins=[cfg.num_cost_bins, cfg.num_score_bins],
            cmap="Greys",
            norm=LogNorm(vmin=1),
            range=[[min_cost, max_cost], [score_min, score_max]],
        )
        fig.colorbar(h[3], ax=ax_density, label="Count", pad=0.02)

        # DP-optimal line
        if band_results:
            band_costs = [r.cost for r in band_results]
            band_scores = [r.score for r in band_results]
            ax_density.plot(
                band_costs, band_scores,
                color="tab:red", lw=2, label="DP optimum", zorder=5,
            )

        # Observed data overlay
        if observed_costs is not None and observed_scores is not None:
            ax_density.scatter(
                observed_costs, observed_scores,
                s=3, alpha=0.3, color="tab:blue", zorder=3,
                label="Observed",
            )

        # Random chance line
        ax_density.axhline(
            y=0.25, color="gray", ls="--", lw=1, alpha=0.7, zorder=2,
        )
        ax_density.text(
            min_cost + 0.02 * (max_cost - min_cost), 0.25 + 0.005,
            "Random chance", fontsize=8, color="gray", va="bottom",
        )

        ax_density.set_ylabel(cfg.score_label)
        ax_density.legend(loc="upper left", fontsize=8)
        ax_density.tick_params(labelbottom=False)

        # ---- Bottom panel: layermap ----
        num_bins = cfg.num_cost_bins
        bin_edges = np.linspace(min_cost, max_cost, num_bins + 1)
        layer_edges = np.arange(num_layers + 1)

        # Build mixer name -> index mapping from phase_order
        mixer_to_idx = {name: i for i, name in enumerate(phase_order)}
        num_mixers = len(phase_order)

        # Build layermap array: (num_layers, num_bins), fill with -1 (empty)
        layermap_data = np.full((num_layers, num_bins), -1, dtype=int)
        for br in band_results:
            # Find which bin this band result belongs to
            b = int(np.digitize(br.cost, bin_edges)) - 1
            b = min(b, num_bins - 1)
            for layer_idx, mixer_name in enumerate(br.assignments):
                layermap_data[layer_idx, b] = mixer_to_idx[mixer_name]

        # Colormap: one color per mixer + white for empty
        from matplotlib.colors import ListedColormap
        cmap_colors = [mixer_colors[name] for name in phase_order] + ["white"]
        mixer_cmap = ListedColormap(cmap_colors)

        ax_layermap.pcolormesh(
            bin_edges, layer_edges, layermap_data,
            cmap=mixer_cmap, vmin=-1, vmax=num_mixers - 1,
        )
        ax_layermap.set_ylabel("Layer index")
        ax_layermap.set_xlabel("Normalized cost")
        ax_layermap.invert_yaxis()

        # Legend for mixer types
        legend_patches = [
            Patch(facecolor=mixer_colors[name], label=name)
            for name in phase_order
        ]
        ax_layermap.legend(
            handles=legend_patches,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=min(len(phase_order), 4),
            fontsize=8,
        )

        fig.tight_layout()

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

    cost_model: CostModelConfig = Field(
        description="Cost model for allocation enumeration and cost computation.",
    )
    exploration_dataset_path: str = Field(
        description="Path to exploration_results.json from explore_placements.",
    )

    # Sampling
    samples_per_allocation: int = Field(
        description="Random permutations per allocation for density estimation.",
        gt=0,
    )
    seed: int = Field(description="Random seed for reproducibility.")

    # Plot
    num_cost_bins: int = Field(
        description="Histogram bins along cost axis.",
        gt=0,
    )
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
