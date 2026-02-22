"""Shared analysis functions for placement landscape commands.

Provides surrogate fitting, cost band computation, and per-band DP solving
used by both placement_landscape and validate_landscape commands.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from place_layers.cost_model import CostModelConfig
from place_layers.cost_model.base_cost_model import Allocation, BaseCostModel
from place_layers.landscape.models import BandResult
from place_layers.mrf.solver import ExactMRFSolver
from place_layers.surrogate.base_surrogate import BaseSurrogate
from place_layers.surrogate.config import SurrogateConfig
from place_layers.surrogate.dataset import ExplorationDataset

logger = logging.getLogger(__name__)


def load_and_fit_surrogate(
    exploration_dataset_path: str,
    cost_model_config: CostModelConfig,
    surrogate_config: SurrogateConfig,
    seed: int,
) -> tuple[ExplorationDataset, BaseSurrogate, BaseCostModel]:
    """Load dataset, create surrogate from config, fit to data.

    Uses the dataset's stored ``feature_config`` directly (the exploration
    phase already selected the best features for this data).

    Args:
        exploration_dataset_path: Path to exploration_results.json.
        cost_model_config: Cost model configuration.
        surrogate_config: Surrogate model configuration (Ridge or Bayesian Ridge).
        seed: Random seed for surrogate fitting.

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

    feature_config = dataset.feature_config
    logger.info(
        "Feature config: order=%d, range=%d, counts=%s",
        feature_config.max_order,
        feature_config.pairwise_max_range,
        feature_config.allocation_counts,
    )

    # Fit surrogate
    scores_array = np.array(dataset.scores)
    surrogate = surrogate_config.create_surrogate(
        mixer_names=dataset.mixer_names, seed=seed, feature_config=feature_config,
    )
    t0 = time.perf_counter()
    surrogate.fit(dataset.assignments, scores_array)
    logger.info("Surrogate fitted (%.3fs)", time.perf_counter() - t0)

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
    surrogate: BaseSurrogate,
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
        surrogate: Fitted surrogate (Ridge or Bayesian Ridge).
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
    t0 = time.perf_counter()
    mrf_model = surrogate.extract_mrf_model(num_layers)
    solver = ExactMRFSolver(mrf_model)
    intercept = mrf_model.intercept
    logger.info("MRF extraction: %.3fs", time.perf_counter() - t0)

    # Solve one unified DP per band
    occupied_bins = sorted(bin_to_allocs.keys())
    best_per_bin: dict[int, BandResult] = {}
    t_dp_start = time.perf_counter()

    for step, b in enumerate(occupied_bins):
        band_allocs = bin_to_allocs[b]
        t0 = time.perf_counter()
        top = solver.solve_top_n(band_allocs, n=1)
        t_band = time.perf_counter() - t0
        logger.info(
            "DP band %d / %d (bin %d, %d allocations, %.3fs)",
            step + 1,
            len(occupied_bins),
            b,
            len(band_allocs),
            t_band,
        )
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

    logger.info(
        "DP complete: %d/%d bands solved in %.3fs",
        len(best_per_bin),
        len(occupied_bins),
        time.perf_counter() - t_dp_start,
    )

    # Sort by cost
    return [best_per_bin[b] for b in sorted(best_per_bin.keys())]
