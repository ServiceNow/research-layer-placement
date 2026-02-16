"""Shared helpers for placement landscape commands.

Extracted from PlacementLandscapeCommand to support both:
- placement_landscape (density plot + DP line)
- validate_landscape (per-band real evaluation)

Pydantic models follow the TraceDataset pattern: model_dump_json / model_validate_json
with .save() / .load() class methods.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from place_layers.cost_model import CostModelConfig
from place_layers.cost_model.base_cost_model import Allocation, BaseCostModel
from place_layers.mrf.solver import ExactMRFSolver
from place_layers.surrogate.dataset import ExplorationDataset
from place_layers.surrogate.model_selection import select_expansion_level
from place_layers.surrogate.ridge_surrogate import RidgeSurrogate

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic models
# ============================================================================


class BandResult(BaseModel):
    """DP-optimal result for one cost band."""

    model_config = {"frozen": True}

    band_index: int
    cost_low: float
    cost_high: float
    cost: float
    score: float
    assignments: tuple[str, ...]


class PlacementResult(BaseModel):
    """A single placement with its cost and score."""

    model_config = {"frozen": True}

    cost: float
    score: float
    assignments: tuple[str, ...]


class BandValidationResult(BaseModel):
    """Per-band surrogate DP + validated search results."""

    model_config = {"frozen": True}

    band_index: int
    cost_low: float
    cost_high: float
    surrogate_dp: PlacementResult
    validated: PlacementResult
    total_evaluations: int


class ValidationSummary(BaseModel):
    """Aggregate statistics across all validated bands."""

    model_config = {"frozen": True}

    spearman_rho: float
    mean_residual: float
    total_bands: int
    total_evaluations: int


class ValidationResults(BaseModel):
    """Full output of validate_landscape command. Serializable to JSON."""

    model_config = {"frozen": True}

    num_layers: int
    baseline_cost: float
    num_cost_bins: int
    cost_bin_edges: list[float]
    bands: list[BandValidationResult]
    summary: ValidationSummary

    def save(self, path: Path) -> None:
        """Save to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> ValidationResults:
        """Load from JSON file."""
        return cls.model_validate_json(path.read_text())


class LandscapeData(BaseModel):
    """Per-band DP results from placement_landscape. Serializable to JSON."""

    model_config = {"frozen": True}

    num_layers: int
    baseline_cost: float
    num_cost_bins: int
    cost_bin_edges: list[float]
    bands: list[BandResult]

    def save(self, path: Path) -> None:
        """Save to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> LandscapeData:
        """Load from JSON file."""
        return cls.model_validate_json(path.read_text())


# ============================================================================
# Shared functions
# ============================================================================


def load_and_fit_surrogate(
    exploration_dataset_path: str,
    cost_model_config: CostModelConfig,
    seed: int,
) -> tuple[ExplorationDataset, RidgeSurrogate, BaseCostModel]:
    """Load dataset, run model selection, fit surrogate.

    Args:
        exploration_dataset_path: Path to exploration_results.json.
        cost_model_config: Cost model configuration.
        seed: Random seed for model selection and surrogate fitting.

    Returns:
        (dataset, surrogate, cost_model) tuple.
    """
    dataset = ExplorationDataset.load(Path(exploration_dataset_path))
    logger.info(
        "Loaded dataset: %d samples, %d layers, rho=%.4f",
        dataset.num_samples,
        dataset.num_layers,
        dataset.final_rho,
    )

    cost_model = cost_model_config.create_cost_model()

    # Model selection
    scores_array = np.array(dataset.scores)
    expansion_result = select_expansion_level(
        assignments=dataset.assignments,
        scores=scores_array,
        mixer_names=dataset.mixer_names,
        seed=seed,
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
        seed=seed,
        feature_config=feature_config,
    )
    surrogate.fit(dataset.assignments, scores_array)
    logger.info("Surrogate fitted")

    return dataset, surrogate, cost_model


def compute_band_edges(
    alloc_costs: np.ndarray,
    baseline_cost: float,
    num_cost_bins: int,
) -> np.ndarray:
    """Return cost bin edges sorted ascending (linearly spaced in speedup-space).

    Args:
        alloc_costs: Array of allocation costs.
        baseline_cost: Maximum cost (1x speedup baseline).
        num_cost_bins: Number of bins.

    Returns:
        Array of (num_cost_bins + 1) cost bin edges, sorted ascending.
    """
    max_speedup = baseline_cost / float(alloc_costs.min())
    speedup_edges = np.linspace(1.0, max_speedup, num_cost_bins + 1)
    return np.sort(baseline_cost / speedup_edges)


def compute_dp_per_band(
    surrogate: RidgeSurrogate,
    all_allocations: list[Allocation],
    alloc_costs: np.ndarray,
    num_layers: int,
    baseline_cost: float,
    cost_model: BaseCostModel,
    num_cost_bins: int,
) -> list[BandResult]:
    """Solve DP per cost band using a single solve_top_n call per band.

    Bins are spaced linearly in speedup-space (speedup = baseline_cost /
    cost) so that display columns map 1-to-1 to DP bands.

    Args:
        surrogate: Fitted RidgeSurrogate.
        all_allocations: All possible allocations.
        alloc_costs: Costs for each allocation.
        num_layers: Number of layers.
        baseline_cost: Maximum cost (1x speedup baseline).
        cost_model: Cost model for cost computation.
        num_cost_bins: Number of cost bins.

    Returns:
        List of BandResult sorted by cost, one per occupied band.
    """
    cost_bin_edges = compute_band_edges(alloc_costs, baseline_cost, num_cost_bins)

    # Assign each allocation to a bin
    bin_indices = np.digitize(alloc_costs, cost_bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_cost_bins - 1)

    # Group allocations by bin
    bin_to_allocs: dict[int, list[Allocation]] = defaultdict(list)
    for i, alloc in enumerate(all_allocations):
        bin_to_allocs[int(bin_indices[i])].append(alloc)

    # Extract MRF model and solver
    mrf_model = surrogate.extract_mrf_model(num_layers)
    solver = ExactMRFSolver(mrf_model)
    intercept = mrf_model.intercept

    # Solve one unified DP per band
    occupied_bins = sorted(bin_to_allocs.keys())
    best_per_bin: dict[int, BandResult] = {}

    for step, b in enumerate(occupied_bins):
        band_allocs = bin_to_allocs[b]
        logger.info(
            "DP band %d / %d (bin %d, %d allocations)",
            step + 1,
            len(occupied_bins),
            b,
            len(band_allocs),
        )
        top = solver.solve_top_n(band_allocs, n=1)
        if not top:
            continue

        solution = top[0]
        score = solution.true_score + intercept
        # Recover cost from winning solution's assignments
        alloc = dict(Counter(solution.placement.assignments))
        cost = cost_model.allocation_cost(alloc)

        # Compute band edges for this bin
        cost_low = float(cost_bin_edges[b])
        cost_high = float(cost_bin_edges[b + 1]) if b + 1 < len(cost_bin_edges) else float(cost_bin_edges[-1])

        best_per_bin[b] = BandResult(
            band_index=b,
            cost_low=cost_low,
            cost_high=cost_high,
            cost=cost,
            score=score,
            assignments=solution.placement.assignments,
        )

    # Sort by cost
    return [best_per_bin[b] for b in sorted(best_per_bin.keys())]
