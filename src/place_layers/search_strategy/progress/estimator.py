"""Search progress estimation.

Estimates how far into a search we are and how much work remains.
Refines estimates as actual evaluation counts replace predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
    budgets: list[int]
    estimated_lower: int
    estimated_upper: int


@dataclass(frozen=True)
class ProgressUpdate:
    """Result of updating progress after an allocation completes.

    Attributes:
        previous: Estimate before the update.
        current: Estimate after the update.
        changed: Whether the estimate actually changed.
    """

    previous: ProgressEstimate
    current: ProgressEstimate
    changed: bool


@dataclass
class SearchProgressEstimator:
    """Estimates search progress and refines as allocations complete.

    Progress = completed_evals + estimate(remaining_allocations)
    As allocations complete, their actual counts replace estimates.

    Attributes:
        initial: Initial estimate before search starts.
        allocation_info: Per-allocation progress info, in exploration order.
        completed_evals: Total evaluations completed so far.
        completed_allocations: Number of allocations fully explored.
    """

    initial: ProgressEstimate
    allocation_info: list[AllocationProgressInfo] = field(default_factory=list)
    completed_evals: int = 0
    completed_allocations: int = 0

    @property
    def current(self) -> ProgressEstimate:
        """Current progress estimate based on completed + remaining."""
        remaining = self.allocation_info[self.completed_allocations :]
        remaining_lower = sum(a.estimated_lower for a in remaining)
        remaining_upper = sum(a.estimated_upper for a in remaining)
        return ProgressEstimate(
            lower=self.completed_evals + remaining_lower,
            upper=self.completed_evals + remaining_upper,
        )

    def allocation_completed(self, actual_evals: int) -> ProgressUpdate:
        """Record an allocation completing with its actual eval count.

        Args:
            actual_evals: Number of evaluations used for this allocation.

        Returns:
            ProgressUpdate with previous and current estimates.
        """
        previous = self.current
        self.completed_allocations += 1
        self.completed_evals += actual_evals
        current = self.current
        return ProgressUpdate(
            previous=previous,
            current=current,
            changed=(previous.lower != current.lower or previous.upper != current.upper),
        )
