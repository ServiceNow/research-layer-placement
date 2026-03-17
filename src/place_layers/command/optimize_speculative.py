"""Optimize speculative decoding: find draft placement minimizing speculative cost.

For speculative decoding with a fixed target placement, the effective
cost of generating tokens is::

    spec_cost = (gamma * c_d + c_t) / (gamma * a_d + 1)

where *gamma* is the draft length, *c_d* is the draft placement cost,
*c_t* is the target placement cost, and *a_d* is the acceptance rate
of the draft placement.

Since *c_d* depends only on the **allocation** (mixer counts) and *a_d*
is maximised within each allocation by the MRF DP solver, the
optimisation decomposes cleanly:

1. For each feasible allocation, DP finds the assignment maximising *a_d*.
2. Compute ``spec_cost`` for each allocation using its optimal *a_d*.
3. Pick the allocation with minimum ``spec_cost``.

Optionally validates the top-k candidates with the real evaluator.

Input
-----
``explore_output_dir`` containing ``exploration_results_{obj}.json``
files (from ``explore_placements`` with an acceptance-rate evaluator).

Output
------
``speculative_results_{obj}.json`` and ``speculative_pareto_{obj}.png``
per objective in the Hydra output dir.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from hydra.core.hydra_config import HydraConfig
from pydantic import BaseModel, Field

from place_layers.backend import BackendConfig
from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.cost_model.base_cost_model import Allocation, BaseCostModel
from place_layers.cost_model.enumeration import enumerate_feasible_allocations
from place_layers.evaluator.evaluators import EvaluatorsConfig
from place_layers.landscape import LandscapeConfig
from place_layers.landscape.config import objective_result_path
from place_layers.mrf.solver import ExactMRFSolver

if TYPE_CHECKING:
    from place_layers.surrogate.base_surrogate import BaseSurrogate

logger = logging.getLogger(__name__)


# ============================================================================
# Result models
# ============================================================================


class SpeculativeCandidate(BaseModel):
    """One candidate draft placement with its speculative decoding cost."""

    model_config = {"frozen": True}

    rank: int
    assignments: tuple[str, ...]
    allocation: dict[str, int]
    draft_cost: float
    surrogate_acceptance_rate: float
    speculative_cost: float
    validated_acceptance_rate: float | None = None
    validated_speculative_cost: float | None = None


class SpeculativeOptimizationResults(BaseModel):
    """Full output of optimize_speculative command."""

    model_config = {"frozen": True}

    objective_name: str
    gamma: int
    target_placement: list[str]
    target_cost: float
    num_layers: int
    num_allocations: int
    candidates: list[SpeculativeCandidate]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> SpeculativeOptimizationResults:
        return cls.model_validate_json(path.read_text())


# ============================================================================
# Helpers
# ============================================================================


def _speculative_cost(
    gamma: int,
    draft_cost: float,
    target_cost: float,
    acceptance_rate: float,
) -> float:
    """Compute speculative decoding cost: (gamma * c_d + c_t) / (gamma * a_d + 1)."""
    return (gamma * draft_cost + target_cost) / (gamma * acceptance_rate + 1)


def _target_cost(
    target_pl_pattern: list[str],
    num_layers: int,
    cost_model: BaseCostModel,
) -> float:
    """Compute target placement cost from the cyclically-extended pattern."""
    target_alloc: dict[str, int] = {}
    for i in range(num_layers):
        mixer = target_pl_pattern[i % len(target_pl_pattern)]
        target_alloc[mixer] = target_alloc.get(mixer, 0) + 1
    return cost_model.allocation_cost(target_alloc)


# ============================================================================
# Command
# ============================================================================


class OptimizeSpeculativeCommand(BaseCommand):
    """Find draft placement minimizing speculative decoding cost."""

    config: "OptimizeSpeculativeCommandConfig"

    async def run(self) -> None:
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)
        logger.info("Output directory: %s", output_dir)

        landscape = self.config.landscape
        explore_dir = Path(landscape.explore_output_dir)

        gamma = self.config.gamma
        target_pl_pattern = self.config.target_placement
        cost_budget = self.config.cost_budget
        logger.info(
            "Speculative params: gamma=%d, target=%s, cost_budget=%s",
            gamma,
            target_pl_pattern,
            f"{cost_budget:.4f}" if cost_budget is not None else "None (all)",
        )

        # Discover objectives from exploration data
        from place_layers.surrogate.dataset import ExplorationDataset

        obj_names = self._discover_objectives(explore_dir)
        first_dataset = ExplorationDataset.load(
            objective_result_path(explore_dir, "exploration_results", obj_names[0]),
        )
        num_layers = first_dataset.num_layers

        # Cost model + target cost
        cost_model = landscape.cost_model.create_cost_model()
        c_t = _target_cost(target_pl_pattern, num_layers, cost_model)
        logger.info("Target placement cost c_t=%.4f", c_t)

        # Enumerate feasible allocations (within cost_budget if set)
        all_allocations = enumerate_feasible_allocations(
            cost_model, num_layers, cost_budget=cost_budget,
        )
        logger.info(
            "Enumerated %d feasible allocations for %d layers",
            len(all_allocations),
            num_layers,
        )

        # Phase 1: surrogate optimisation per objective
        per_objective: dict[str, list[SpeculativeCandidate]] = {}
        per_objective_surrogate: dict[str, BaseSurrogate] = {}

        for obj_name in obj_names:
            candidates, surrogate = self._surrogate_optimize(
                obj_name=obj_name,
                explore_dir=explore_dir,
                cost_model=cost_model,
                all_allocations=all_allocations,
                num_layers=num_layers,
                gamma=gamma,
                c_t=c_t,
            )
            per_objective[obj_name] = candidates
            per_objective_surrogate[obj_name] = surrogate

        # Phase 2: optional validation with real evaluator
        if self.config.validate_top_k > 0:
            if self.config.evaluators is None:
                raise ValueError(
                    "evaluators config is required when validate_top_k > 0"
                )
            await self._validate(
                per_objective=per_objective,
                output_dir=output_dir,
                gamma=gamma,
                c_t=c_t,
            )

        # Save results and plots
        for obj_name, candidates in per_objective.items():
            results = SpeculativeOptimizationResults(
                objective_name=obj_name,
                gamma=gamma,
                target_placement=target_pl_pattern,
                target_cost=c_t,
                num_layers=num_layers,
                num_allocations=len(all_allocations),
                candidates=candidates,
            )
            results_path = objective_result_path(
                output_dir, "speculative_results", obj_name,
            )
            results.save(results_path)
            logger.info("Saved results for '%s' -> %s", obj_name, results_path)

            try:
                plot_path = output_dir / f"speculative_pareto_{obj_name}.png"
                self._plot_pareto(
                    candidates, gamma, c_t, obj_name, plot_path,
                    phase_order=cost_model.phase_order,
                    surrogate=per_objective_surrogate.get(obj_name),
                )
            except ImportError:
                logger.warning(
                    "matplotlib not installed — skipping Pareto plot"
                )
            except Exception:
                logger.warning(
                    "Failed to generate Pareto plot", exc_info=True
                )

    # -- Helpers ----------------------------------------------------------------

    @staticmethod
    def _discover_objectives(explore_dir: Path) -> list[str]:
        """Find objective names from the exploration output directory.

        Supports two conventions:
        - New: {explore_dir}/{obj_name}/exploration_results.json
        - Legacy: {explore_dir}/exploration_results_{obj_name}.json
        """
        names: list[str] = []

        # New convention: subdirectory per objective
        for p in sorted(explore_dir.glob("*/exploration_results.json")):
            names.append(p.parent.name)

        # Legacy convention: flat files with objective name suffix
        if not names:
            prefix = "exploration_results_"
            for p in sorted(explore_dir.glob(f"{prefix}*.json")):
                names.append(p.stem[len(prefix):])

        if not names:
            raise FileNotFoundError(
                f"No exploration results found in {explore_dir}. "
                f"Expected either {explore_dir}/<obj>/exploration_results.json "
                f"or {explore_dir}/exploration_results_<obj>.json."
            )
        logger.info("Discovered objectives: %s", names)
        return names

    # -- Phase 1: surrogate DP --------------------------------------------------

    def _surrogate_optimize(
        self,
        *,
        obj_name: str,
        explore_dir: Path,
        cost_model: BaseCostModel,
        all_allocations: list[Allocation],
        num_layers: int,
        gamma: int,
        c_t: float,
    ) -> tuple[list[SpeculativeCandidate], BaseSurrogate]:
        """Fit surrogate, DP-solve per allocation, rank by speculative cost.

        Returns:
            (candidates, surrogate) — surrogate is needed for uncertainty bands.
        """
        landscape = self.config.landscape
        logger.info("=== Surrogate optimisation: %s ===", obj_name)

        import numpy as np

        from place_layers.surrogate.config import BayesianRidgeSurrogateConfig
        from place_layers.surrogate.dataset import ExplorationDataset
        from place_layers.surrogate.evaluation_log import (
            PooledEvaluationData,
            discover_evaluation_records,
        )

        dataset_path = objective_result_path(
            explore_dir, "exploration_results", obj_name,
        )
        dataset = ExplorationDataset.load(dataset_path)

        pooled = PooledEvaluationData.from_exploration(dataset)
        supplemental_files = (
            discover_evaluation_records(
                list(landscape.supplemental_data_dirs), dataset.objective_name,
                num_layers=dataset.num_layers,
                mixer_names=list(dataset.mixer_names),
            )
            if landscape.supplemental_data_dirs else []
        )
        if supplemental_files:
            pooled = pooled.merge(PooledEvaluationData.from_jsonl(supplemental_files))

        surrogate = landscape.surrogate.create_surrogate(
            mixer_names=list(dataset.mixer_names), seed=landscape.seed,
            feature_config=dataset.feature_config,
        )
        match landscape.surrogate:
            case BayesianRidgeSurrogateConfig(alpha_selection="evidence"):
                selected_alpha = surrogate.fit_optimize_alpha(
                    list(pooled.assignments), np.array(pooled.scores),
                )
                logger.info("Type-II ML alpha: %.4g", selected_alpha)
            case _:
                surrogate.fit(list(pooled.assignments), np.array(pooled.scores))

        # MRF extraction + per-allocation DP
        t0 = time.perf_counter()
        mrf_model = surrogate.extract_mrf_model(num_layers)
        solver = ExactMRFSolver(mrf_model)
        intercept = mrf_model.intercept

        per_alloc = solver.solve_top_n_per_allocation(
            all_allocations, n_per_allocation=1,
        )
        logger.info(
            "DP solved %d allocations in %.3fs",
            len(per_alloc),
            time.perf_counter() - t0,
        )

        # Build candidates
        candidates: list[SpeculativeCandidate] = []
        for _counts_tuple, solutions in per_alloc.items():
            sol = solutions[0]
            acceptance_rate = max(0.0, sol.true_score + intercept)
            alloc_dict = dict(Counter(sol.placement.assignments))
            draft_cost = cost_model.allocation_cost(alloc_dict)
            spec_cost = _speculative_cost(gamma, draft_cost, c_t, acceptance_rate)
            candidates.append(
                SpeculativeCandidate(
                    rank=0,
                    assignments=sol.placement.assignments,
                    allocation=alloc_dict,
                    draft_cost=draft_cost,
                    surrogate_acceptance_rate=acceptance_rate,
                    speculative_cost=spec_cost,
                )
            )

        # Sort by speculative cost ascending (cheapest first) and assign ranks
        candidates.sort(key=lambda c: c.speculative_cost)
        candidates = [
            c.model_copy(update={"rank": i + 1}) for i, c in enumerate(candidates)
        ]

        # Log top results
        n_show = min(10, len(candidates))
        logger.info(
            "Top %d candidates by speculative cost (gamma=%d, c_t=%.4f):",
            n_show,
            gamma,
            c_t,
        )
        for c in candidates[:n_show]:
            logger.info(
                "  #%d: spec=%.4f  c_d=%.4f  a_d=%.4f  alloc=%s",
                c.rank,
                c.speculative_cost,
                c.draft_cost,
                c.surrogate_acceptance_rate,
                c.allocation,
            )

        return candidates, surrogate

    # -- Phase 2: validation with real evaluator --------------------------------

    async def _validate(
        self,
        *,
        per_objective: dict[str, list[SpeculativeCandidate]],
        output_dir: Path,
        gamma: int,
        c_t: float,
    ) -> None:
        """Validate top-k candidates with real evaluator (in-place update)."""
        from contextlib import nullcontext

        from place_layers.core.types import LayerPlacement

        backend_cm = (
            self.config.backend.create_backend()
            if self.config.backend
            else nullcontext()
        )
        async with backend_cm as backend:
            async with self.config.evaluators.create_evaluators(
                output_dir, backend=backend,
            ) as evals:
                for obj_name, candidates in per_objective.items():
                    top_k = candidates[: self.config.validate_top_k]
                    logger.info(
                        "Validating top %d candidates for '%s'...",
                        len(top_k),
                        obj_name,
                    )
                    validated: list[SpeculativeCandidate] = []
                    for c in top_k:
                        placement = LayerPlacement(assignments=c.assignments)
                        obj_scores = await evals.evaluate_objectives(
                            placement, names=[obj_name],
                        )
                        real_ar = obj_scores[obj_name]
                        real_spec = _speculative_cost(
                            gamma, c.draft_cost, c_t, real_ar,
                        )
                        validated.append(
                            c.model_copy(
                                update={
                                    "validated_acceptance_rate": real_ar,
                                    "validated_speculative_cost": real_spec,
                                }
                            )
                        )
                        logger.info(
                            "  #%d: surr_ar=%.4f  real_ar=%.4f  "
                            "surr_spec=%.4f  real_spec=%.4f",
                            c.rank,
                            c.surrogate_acceptance_rate,
                            real_ar,
                            c.speculative_cost,
                            real_spec,
                        )
                    per_objective[obj_name] = (
                        validated + candidates[self.config.validate_top_k :]
                    )

    # -- Phase 3: Pareto frontier visualisation ---------------------------------

    @staticmethod
    def _plot_pareto(
        candidates: list[SpeculativeCandidate],
        gamma: int,
        c_t: float,
        obj_name: str,
        output_path: Path,
        *,
        phase_order: list[str],
        surrogate: BaseSurrogate | None = None,
    ) -> None:
        """Save a two-panel figure: Pareto scatter (top) + layermap (bottom)."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import BoundaryNorm, ListedColormap
        from matplotlib.patches import Patch

        draft_costs = np.array([c.draft_cost for c in candidates])
        spec_costs = np.array([c.speculative_cost for c in candidates])
        acceptance_rates = np.array(
            [c.surrogate_acceptance_rate for c in candidates]
        )

        order = np.argsort(draft_costs)
        num_layers = len(candidates[0].assignments)

        # Use cost model's phase_order for consistent mixer colors
        tab_colors = [
            "tab:blue", "tab:orange", "tab:green", "tab:red",
            "tab:purple", "tab:brown", "tab:pink", "tab:gray",
        ]
        mixer_colors = {
            name: tab_colors[i % len(tab_colors)]
            for i, name in enumerate(phase_order)
        }
        mixer_to_idx = {name: i for i, name in enumerate(phase_order)}

        # ── Two-panel layout ─────────────────────────────────────────────
        fig = plt.figure(figsize=(10, 8), dpi=150)
        gs = fig.add_gridspec(
            2, 2, height_ratios=[3, 1], width_ratios=[1, 0.03],
            hspace=0.08, wspace=0.05,
        )
        ax_scatter = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        ax_layermap = fig.add_subplot(gs[1, 0], sharex=ax_scatter)

        # ── Top panel: Pareto scatter ────────────────────────────────────
        sc = ax_scatter.scatter(
            draft_costs,
            spec_costs,
            c=acceptance_rates,
            cmap="viridis",
            s=30,
            alpha=0.7,
            edgecolors="none",
            zorder=2,
        )
        fig.colorbar(sc, cax=cax, label="Surrogate acceptance rate")

        sorted_dc = draft_costs[order]
        sorted_sc = spec_costs[order]
        sorted_ar = acceptance_rates[order]
        window = 5
        half = window // 2
        n = len(order)

        # For each position find the argmin within the rolling window
        win_argmin = np.array([
            int(np.argmin(sorted_sc[max(0, i - half):min(n, i + half + 1)]))
            + max(0, i - half)
            for i in range(n)
        ])
        smooth_sc = sorted_sc[win_argmin]

        ax_scatter.plot(
            sorted_dc, smooth_sc,
            color="tab:red", linewidth=1.5, alpha=0.8,
            label="Best per cost (smoothed)", zorder=3,
        )

        # Uncertainty band: propagate sigma of the selected candidate at each position
        if surrogate is not None and len(candidates) >= 2:
            all_assignments = [candidates[i].assignments for i in order]
            _, sigma_ar = surrogate.predict(all_assignments)

            spec_lo = np.array([
                _speculative_cost(
                    gamma, sorted_dc[i], c_t,
                    max(1e-6, sorted_ar[win_argmin[i]] + 2 * sigma_ar[win_argmin[i]])
                )
                for i in range(n)
            ])
            spec_hi = np.array([
                _speculative_cost(
                    gamma, sorted_dc[i], c_t,
                    max(1e-6, sorted_ar[win_argmin[i]] - 2 * sigma_ar[win_argmin[i]])
                )
                for i in range(n)
            ])

            ax_scatter.fill_between(
                sorted_dc, spec_lo, spec_hi,
                alpha=0.15, color="tab:red",
                label="Posterior \u00b12\u03c3", zorder=1,
            )

        best = candidates[0]
        ax_scatter.scatter(
            [best.draft_cost], [best.speculative_cost],
            marker="*", s=250, c="tab:red",
            edgecolors="black", linewidths=0.8, zorder=5,
            label=f"Optimum (c_d={best.draft_cost:.2f}, "
            f"a_d={best.surrogate_acceptance_rate:.3f})",
        )

        baseline_spec = _speculative_cost(gamma, c_t, c_t, 1.0)
        ax_scatter.axhline(
            baseline_spec, color="grey", linestyle="--",
            linewidth=1, alpha=0.6,
            label=f"Target-only baseline ({baseline_spec:.2f})", zorder=1,
        )

        validated = [
            c for c in candidates if c.validated_speculative_cost is not None
        ]
        if validated:
            for c in validated:
                ax_scatter.plot(
                    [c.draft_cost, c.draft_cost],
                    [c.speculative_cost, c.validated_speculative_cost],
                    color="tab:orange", linewidth=0.8, alpha=0.6, zorder=4,
                )
            ax_scatter.scatter(
                [c.draft_cost for c in validated],
                [c.validated_speculative_cost for c in validated],
                marker="D", s=50, c="tab:orange",
                edgecolors="black", linewidths=0.6, zorder=5,
                label="Validated",
            )

        ax_scatter.set_ylabel("Speculative cost", fontsize=11)
        ax_scatter.set_title(
            f"Speculative cost ({obj_name}, "
            fr"$\gamma$={gamma}, c_t={c_t:.2f})",
            fontsize=12,
        )
        ax_scatter.legend(fontsize=9, loc="upper right")
        ax_scatter.grid(True, alpha=0.3)
        ax_scatter.tick_params(labelbottom=False)

        # ── Bottom panel: layermap showing best placement at each cost ────
        # For each cost bin, show the placement of the candidate with the
        # lowest speculative cost whose draft cost falls in that bin.
        cost_min = float(draft_costs.min())
        cost_max = float(draft_costs.max())
        num_bins = max(len(candidates), 40)
        bin_edges = np.linspace(cost_min, cost_max, num_bins + 1)

        layermap = np.full((num_layers, num_bins), -1, dtype=int)

        # Group candidates into bins; pick the one with lowest spec cost per bin
        for c in candidates:
            b = int(np.searchsorted(bin_edges[1:], c.draft_cost, side="left"))
            b = min(b, num_bins - 1)
            if layermap[0, b] == -1:
                for layer, mixer in enumerate(c.assignments):
                    layermap[layer, b] = mixer_to_idx[mixer]

        # Forward-fill empty bins from the nearest filled neighbor
        for b in range(1, num_bins):
            if layermap[0, b] == -1:
                layermap[:, b] = layermap[:, b - 1]
        # Back-fill any leading empty bins
        for b in range(num_bins - 2, -1, -1):
            if layermap[0, b] == -1:
                layermap[:, b] = layermap[:, b + 1]

        layer_edges = np.arange(num_layers + 1)
        cmap_colors = ["white"] + [mixer_colors[name] for name in phase_order]
        mixer_cmap = ListedColormap(cmap_colors)
        boundaries = np.arange(-1.5, len(phase_order) + 0.5, 1)
        norm = BoundaryNorm(boundaries, len(cmap_colors))

        ax_layermap.pcolormesh(
            bin_edges, layer_edges, layermap,
            cmap=mixer_cmap, norm=norm,
        )
        ax_layermap.set_ylabel("Layer index", fontsize=11)
        ax_layermap.set_xlabel("Draft cost (c_d)", fontsize=11)
        ax_layermap.invert_yaxis()

        legend_patches = [
            Patch(facecolor=mixer_colors[name], label=name)
            for name in phase_order
        ]
        ax_layermap.legend(
            handles=legend_patches,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=min(len(phase_order), 4),
            fontsize=9,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved Pareto plot -> %s", output_path)


# ============================================================================
# Config
# ============================================================================


class OptimizeSpeculativeCommandConfig(BaseCommandConfig):
    """Configuration for optimize_speculative command.

    Finds the draft placement minimising speculative decoding cost
    ``(gamma * c_d + c_t) / (gamma * a_d + 1)`` via surrogate-guided
    DP optimisation.
    """

    _impl_class: ClassVar[type[BaseCommand]] = OptimizeSpeculativeCommand

    command: Literal["optimize_speculative"] = "optimize_speculative"

    gamma: int = Field(
        gt=0,
        description="Draft length (number of speculative tokens per step).",
    )
    target_placement: list[str] = Field(
        description="Target placement pattern (cyclically extended to num_layers). "
        "Used to compute target cost c_t.",
    )
    cost_budget: float | None = Field(
        default=None,
        gt=0,
        description="Maximum draft cost to consider. Should match the budget "
        "used during exploration so the surrogate stays in-distribution. "
        "None = enumerate all allocations.",
    )

    landscape: LandscapeConfig = Field(
        description="Shared landscape configuration (cost model, surrogate, "
        "explore output dir).",
    )

    backend: BackendConfig | None = Field(
        default=None,
        description="Backend configuration. Required when validate_top_k > 0.",
    )
    evaluators: EvaluatorsConfig | None = Field(
        default=None,
        description="Evaluator configuration. Required when validate_top_k > 0.",
    )

    validate_top_k: int = Field(
        default=0,
        ge=0,
        description="Number of top candidates to validate with real evaluator. "
        "0 = surrogate-only (no real evaluation needed).",
    )
