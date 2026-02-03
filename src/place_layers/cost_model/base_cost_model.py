"""Base cost model for layer placement optimization."""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Literal

from place_layers.core.types import LayerPlacement


# Type alias for allocation: counts per mixer (length = num_mixers)
# Example: (2, 1, 3) means 2 layers on mixer 0, 1 on mixer 1, 3 on mixer 2
Allocation = tuple[int, ...]


class BaseCostModel(ABC):
    """
    Abstract base class for cost models.

    A cost model computes the "cost" of a layer placement configuration.
    This is typically related to inference throughput/latency, where different
    mixer types have different computational costs.

    The cost function should be:
    - Deterministic: same configuration always produces same cost
    - Decomposable: ideally, cost changes from single-layer swaps are O(1)

    Cost models also provide navigation over the allocation space (counts per
    mixer). This enables efficient search algorithms that explore allocations
    in cost order without enumerating all possibilities.

    Terminology:
    - LayerPlacement: which mixer each layer uses (length = num_layers)
    - Allocation: how many layers use each mixer (length = num_mixers)

    Cost depends only on allocation (counts), not on which specific layers.
    Performance depends on the full placement.
    """

    @property
    @abstractmethod
    def num_mixers(self) -> int:
        """Number of mixer types."""
        pass

    @property
    @abstractmethod
    def phase_order(self) -> list[str]:
        """
        Mixer names in phase order for hierarchical search.

        This determines the order in which phases assign layers to mixers.
        The first mixer in the list is assigned first, then the second, etc.
        The last mixer in the list receives all remaining layers.

        Implementations should return mixer names in the order specified
        by the cost model configuration (e.g., the key order in mixer_costs dict).
        """
        pass

    # =========================================================================
    # Assignment-level operations
    # =========================================================================

    @abstractmethod
    def cost(self, placement: LayerPlacement) -> float:
        """
        Compute the cost of a layer placement configuration.

        Args:
            placement: The layer-to-mixer assignment.

        Returns:
            Cost value (lower is cheaper/faster).
        """
        pass

    @abstractmethod
    def cost_delta(
        self, layer_idx: int, old_mixer: str, new_mixer: str
    ) -> float:
        """
        Compute the change in cost from swapping a single layer's mixer.

        This enables efficient incremental cost updates during search.

        Args:
            layer_idx: Index of the layer being changed.
            old_mixer: Current mixer name for the layer.
            new_mixer: New mixer name for the layer.

        Returns:
            Change in cost (positive = more expensive, negative = cheaper).
        """
        pass

    @abstractmethod
    def min_cost(self, num_layers: int) -> float:
        """
        Compute the minimum possible cost (all layers on cheapest mixer).

        Args:
            num_layers: Number of layers in the model.

        Returns:
            Minimum achievable cost.
        """
        pass

    @abstractmethod
    def max_cost(self, num_layers: int) -> float:
        """
        Compute the maximum possible cost (all layers on most expensive mixer).

        Args:
            num_layers: Number of layers in the model.

        Returns:
            Maximum possible cost.
        """
        pass

    # =========================================================================
    # Allocation-level operations
    # =========================================================================

    @abstractmethod
    def allocation_cost(self, allocation: Allocation) -> float:
        """
        Compute cost of an allocation (counts per mixer).

        Args:
            allocation: Tuple of (n_0, n_1, ..., n_{K-1}) where n_i is
                       the number of layers using mixer i.

        Returns:
            Cost value for this allocation.
        """
        pass

    @abstractmethod
    def min_cost_allocation(self, num_layers: int) -> Allocation:
        """
        Return the allocation with minimum cost.

        Args:
            num_layers: Total number of layers (sum of allocation).

        Returns:
            Allocation tuple with all layers on cheapest mixer(s).
        """
        pass

    @abstractmethod
    def max_cost_allocation(self, num_layers: int) -> Allocation:
        """
        Return the allocation with maximum cost.

        Args:
            num_layers: Total number of layers (sum of allocation).

        Returns:
            Allocation tuple with all layers on most expensive mixer(s).
        """
        pass

    @abstractmethod
    def allocation_at_cost_bound(
        self,
        num_layers: int,
        cost_bound: float,
        from_above: bool = True,
    ) -> Allocation | None:
        """
        Find allocation closest to a cost bound.

        Args:
            num_layers: Total number of layers.
            cost_bound: Target cost value.
            from_above: If True, find largest cost <= bound.
                       If False, find smallest cost >= bound.

        Returns:
            Allocation closest to the bound, or None if infeasible.

        Complexity:
            Linear cost models: O(K log K) exact
            Non-linear cost models: may be approximate
        """
        pass

    @abstractmethod
    def allocation_neighbors(
        self,
        allocation: Allocation,
        direction: Literal["increasing", "decreasing"],
    ) -> Iterator[tuple[Allocation, float]]:
        """
        Yield neighboring allocations in cost order.

        A neighbor differs by moving one layer from one mixer to another.
        Neighbors are yielded in order of cost change (delta).

        Args:
            allocation: Current allocation.
            direction: "increasing" for neighbors with higher cost,
                      "decreasing" for neighbors with lower cost.

        Yields:
            (neighbor_allocation, cost_delta) pairs, sorted by |delta|.

        Complexity:
            Linear cost models: O(K²) exact ordering
            Non-linear cost models: may be approximate ordering
        """
        pass

    def count_allocations(self, num_layers: int) -> int:
        """
        Count total number of possible allocations (stars-and-bars formula).

        The number of ways to distribute num_layers layers among num_mixers mixers
        is C(num_layers + num_mixers - 1, num_mixers - 1).

        Args:
            num_layers: Total number of layers.

        Returns:
            Total number of distinct allocations.
        """
        from math import comb

        return comb(num_layers + self.num_mixers - 1, self.num_mixers - 1)

    def allocations_by_cost(
        self,
        num_layers: int,
        direction: Literal["increasing", "decreasing"] = "increasing",
    ) -> Iterator[tuple[float, Allocation]]:
        """
        Iterate over all allocations in cost order.

        Uses a heap-based traversal that visits each allocation exactly once.

        Args:
            num_layers: Total number of layers.
            direction: "increasing" starts from min cost,
                      "decreasing" starts from max cost.

        Yields:
            (cost, allocation) pairs in the specified order.

        Complexity:
            O(A * K² * log A) where A = number of allocations
            A = C(N+K-1, K-1) for N layers and K mixers
        """
        if direction == "increasing":
            start = self.min_cost_allocation(num_layers)
            neighbor_direction: Literal["increasing", "decreasing"] = "increasing"
            heap_sign = 1  # Min-heap
        else:
            start = self.max_cost_allocation(num_layers)
            neighbor_direction = "decreasing"
            heap_sign = -1  # Max-heap via negation

        start_cost = self.allocation_cost(start)
        frontier: list[tuple[float, Allocation]] = [(heap_sign * start_cost, start)]
        visited: set[Allocation] = {start}

        while frontier:
            signed_cost, alloc = heapq.heappop(frontier)
            cost = heap_sign * signed_cost
            yield cost, alloc

            for neighbor, _delta in self.allocation_neighbors(alloc, neighbor_direction):
                if neighbor not in visited:
                    visited.add(neighbor)
                    neighbor_cost = self.allocation_cost(neighbor)
                    heapq.heappush(frontier, (heap_sign * neighbor_cost, neighbor))
