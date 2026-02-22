"""Shared analysis functions for placement landscape commands.

Provides surrogate fitting, cost band computation, and per-band DP solving
used by both placement_landscape and validate_landscape commands.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np

from place_layers.cost_model import CostModelConfig
from place_layers.cost_model.base_cost_model import Allocation, BaseCostModel
from place_layers.landscape.models import BandResult
from place_layers.mrf.solver import ExactMRFSolver, ExactSolution
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
    """Solve DP for all cost bands, using single pass when feasible.

    Tries a single DP build over all allocations (fast when K <= 3).
    If the combined state space is too large (K >= 4 with large N),
    falls back to one DP build per cost band.

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

    # Extract MRF model and solver
    t0 = time.perf_counter()
    mrf_model = surrogate.extract_mrf_model(num_layers)
    solver = ExactMRFSolver(mrf_model)
    intercept = mrf_model.intercept
    logger.info("MRF extraction: %.3fs", time.perf_counter() - t0)

    mixer_names = mrf_model.mixer_names

    # Map counts_tuple -> band index
    counts_to_bin: dict[tuple[int, ...], int] = {}
    for i, alloc in enumerate(all_allocations):
        counts = tuple(alloc.get(name, 0) for name in mixer_names)
        counts_to_bin[counts] = int(bin_indices[i])

    # Single call handles both fast-path (single DP build) and fallback
    # (recursive bisection) internally.
    t_dp_start = time.perf_counter()
    per_alloc = solver.solve_top_n_per_allocation(all_allocations, n_per_allocation=1)

    best_per_bin: dict[int, tuple[float, ExactSolution]] = {}
    for counts_tuple, sols in per_alloc.items():
        b = counts_to_bin.get(counts_tuple)
        if b is None:
            continue
        sol = sols[0]
        prev = best_per_bin.get(b)
        if prev is None or sol.true_score > prev[0]:
            best_per_bin[b] = (sol.true_score, sol)

    logger.info(
        "DP complete: %d bands solved in %.3fs (%d allocations)",
        len(best_per_bin),
        time.perf_counter() - t_dp_start,
        len(all_allocations),
    )

    # Build BandResult objects
    results: dict[int, BandResult] = {}
    for b, (_, solution) in best_per_bin.items():
        score = solution.true_score + intercept
        alloc = dict(Counter(solution.placement.assignments))
        cost = cost_model.allocation_cost(alloc)

        cost_low = float(cost_bin_edges[b])
        cost_high = float(cost_bin_edges[b + 1]) if b + 1 < len(cost_bin_edges) else float(cost_bin_edges[-1])

        results[b] = BandResult(
            band_index=b,
            cost_low=cost_low,
            cost_high=cost_high,
            cost=cost,
            score=score,
            assignments=solution.placement.assignments,
        )

    # Sort by cost
    return [results[b] for b in sorted(results.keys())]
