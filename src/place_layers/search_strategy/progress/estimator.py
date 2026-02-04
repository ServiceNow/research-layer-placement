"""Search progress estimation.

Stateless estimator for search progress bounds. The estimator is a lookup table
that computes bounds given the caller's current progress state, without maintaining
any mutable state itself. This makes it safe to reuse across multiple searches.

Statistical Bounds Approach:
---------------------------
Instead of worst-case bounds (which give 8x spread), we model early stopping
probabilistically. Each growth step is a Bernoulli random variable - it either
runs or doesn't, with P(run step s) = continuation_rate^s.

This gives us:
- Expected value: Best single estimate
- Variance: How uncertain we are
- Statistical bounds: expected ± 2*std_dev (95% CI)

As allocations complete, variance decreases and bounds tighten automatically.
"""

from __future__ import annotations

import math

from pydantic import BaseModel


class ProgressEstimate(BaseModel):
    """Estimated evaluation count range with statistical bounds.

    Uses expected value and variance from probabilistic early stopping model.
    Bounds are computed as expected ± 2*std_dev (95% confidence interval).

    Attributes:
        lower: Statistical lower bound (expected - 2*std_dev).
        upper: Statistical upper bound (expected + 2*std_dev).
        expected: Expected value (best single estimate).
        std_dev: Standard deviation for confidence intervals.
        confidence: Confidence level for bounds (default 0.95).
    """

    model_config = {"frozen": True}

    lower: int
    upper: int
    expected: int = 0
    std_dev: float = 0.0
    confidence: float = 0.95


class AllocationProgressInfo(BaseModel):
    """Progress info for a single allocation.

    Used by hierarchical search to track per-allocation statistics for
    progressive refinement.

    Attributes:
        allocation: The allocation tuple (counts per mixer).
        budgets: Phase budgets (layers to assign per phase).
        expected: Expected evaluations for this allocation.
        variance: Variance in evaluations (for aggregation).
        estimated_lower: Statistical lower bound (expected - 2*sqrt(variance)).
        estimated_upper: Statistical upper bound (expected + 2*sqrt(variance)).
    """

    model_config = {"frozen": True}

    allocation: tuple[int, ...]
    budgets: tuple[int, ...]
    expected: float
    variance: float
    estimated_lower: int
    estimated_upper: int


class SearchProgressEstimator(BaseModel):
    """Stateless progress estimator - a lookup table for evaluation bounds.

    This is an immutable data structure that provides bounds calculations
    based on caller-provided state. It does not track any mutable state,
    making it safe to reuse across multiple searches or search strategies.

    The caller (search strategy) maintains progress state (completed_allocations,
    completed_evals) and queries this estimator for bounds.

    Attributes:
        initial: Initial estimate before search starts.
        allocation_info: Per-allocation progress info, in exploration order.
    """

    model_config = {"frozen": True}

    initial: ProgressEstimate
    allocation_info: tuple[AllocationProgressInfo, ...]

    def bounds_after(
        self,
        completed_allocations: int,
        completed_evals: int,
    ) -> ProgressEstimate:
        """Get statistical bounds given current progress state.

        Uses expected values and variances from remaining allocations.
        Bounds tighten as allocations complete (variance decreases).

        Args:
            completed_allocations: Number of allocations fully explored.
            completed_evals: Total evaluations completed so far.

        Returns:
            ProgressEstimate with statistical lower/upper bounds and expected value.
        """
        remaining = self.allocation_info[completed_allocations:]

        # Sum expectations and variances (variances add for independent RVs)
        remaining_expected = sum(a.expected for a in remaining)
        remaining_variance = sum(a.variance for a in remaining)
        remaining_std_dev = math.sqrt(remaining_variance) if remaining_variance > 0 else 0.0

        total_expected = completed_evals + remaining_expected

        # 95% confidence interval: expected ± 2*std_dev
        return ProgressEstimate(
            lower=int(total_expected - 2 * remaining_std_dev),
            upper=int(total_expected + 2 * remaining_std_dev),
            expected=int(total_expected),
            std_dev=remaining_std_dev,
        )
