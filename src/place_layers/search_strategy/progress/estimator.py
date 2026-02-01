"""Search progress estimation.

Stateless estimator for search progress bounds. The estimator is a lookup table
that computes bounds given the caller's current progress state, without maintaining
any mutable state itself. This makes it safe to reuse across multiple searches.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressEstimate:
    """Estimated evaluation count range.

    Attributes:
        lower: Lower bound on evaluation count.
        upper: Upper bound on evaluation count.
        confidence: 1.0 = exact bounds, <1.0 = approximate (e.g., due to early stopping).
    """

    lower: int
    upper: int
    confidence: float = 1.0


@dataclass(frozen=True)
class AllocationProgressInfo:
    """Progress info for a single allocation.

    Used by hierarchical search to track per-allocation bounds for
    progressive refinement.

    Attributes:
        allocation: The allocation tuple (counts per mixer).
        budgets: Phase budgets (layers to assign per phase).
        estimated_lower: Lower bound estimate for this allocation.
        estimated_upper: Upper bound estimate for this allocation.
    """

    allocation: tuple[int, ...]
    budgets: tuple[int, ...]
    estimated_lower: int
    estimated_upper: int


@dataclass(frozen=True)
class SearchProgressEstimator:
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

    initial: ProgressEstimate
    allocation_info: tuple[AllocationProgressInfo, ...]

    def bounds_after(
        self,
        completed_allocations: int,
        completed_evals: int,
    ) -> ProgressEstimate:
        """Get bounds given current progress state.

        Computes: completed_evals + estimate(remaining_allocations)

        Args:
            completed_allocations: Number of allocations fully explored.
            completed_evals: Total evaluations completed so far.

        Returns:
            ProgressEstimate with lower and upper bounds.
        """
        remaining = self.allocation_info[completed_allocations:]
        remaining_lower = sum(a.estimated_lower for a in remaining)
        remaining_upper = sum(a.estimated_upper for a in remaining)
        return ProgressEstimate(
            lower=completed_evals + remaining_lower,
            upper=completed_evals + remaining_upper,
        )
