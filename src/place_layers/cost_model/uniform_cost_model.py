"""Uniform cost model: all placements have equal (zero) cost.

Use this when the optimization problem has no cost constraint and all
mixer types are equally "expensive".  The inherited BFS in
``allocations_by_cost`` still works because ``allocation_neighbors``
returns all same-cost neighbors (delta=0 passes the strict-inequality
filter in the base class).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from place_layers.core.types import LayerPlacement
from place_layers.cost_model.base_cost_model import Allocation, BaseCostModel
from place_layers.cost_model.base_cost_model_config import BaseCostModelConfig


class UniformCostModel(BaseCostModel):
    """Cost model where every placement has cost 0.

    This is the trivial cost model: no mixer is more expensive than any
    other.  It is useful for problems that do not have a cost constraint
    but still need a cost model object (e.g. the explore command).
    """

    def __init__(self, config: "UniformCostModelConfig") -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Assignment-level operations
    # ------------------------------------------------------------------

    def cost(self, placement: LayerPlacement) -> float:
        return 0.0

    def cost_delta(self, layer_idx: int, old_mixer: str, new_mixer: str) -> float:
        return 0.0

    def min_cost(self, num_layers: int) -> float:
        return 0.0

    def max_cost(self, num_layers: int) -> float:
        return 0.0

    # ------------------------------------------------------------------
    # Allocation-level operations
    # ------------------------------------------------------------------

    def allocation_cost(self, allocation: Allocation) -> float:
        return 0.0

    def min_cost_allocation(self, num_layers: int) -> Allocation:
        # All layers on the first mixer (arbitrary but deterministic).
        first = self.config.mixer_names[0]
        return {
            name: (num_layers if name == first else 0)
            for name in self.config.mixer_names
        }

    def max_cost_allocation(self, num_layers: int) -> Allocation:
        # Same as min — all allocations have cost 0.
        return self.min_cost_allocation(num_layers)

    def allocation_at_cost_bound(
        self,
        num_layers: int,
        cost_bound: float,
        from_above: bool = True,
    ) -> Allocation | None:
        # Every allocation satisfies any non-negative bound.
        if cost_bound < 0:
            return None
        return self.min_cost_allocation(num_layers)

    def allocation_neighbors(
        self,
        allocation: Allocation,
        direction: Literal["increasing", "decreasing"],
    ) -> Iterator[tuple[Allocation, float]]:
        """Yield all single-swap neighbors, each with delta=0.

        Since all costs are equal, the direction filter in the base BFS
        uses strict inequality (``delta < 0`` / ``delta > 0``), so
        delta=0 moves pass in *both* directions — exactly what we need
        for the BFS to discover all allocations.
        """
        for from_mixer, count in allocation.items():
            if count == 0:
                continue
            for to_mixer in allocation:
                if from_mixer == to_mixer:
                    continue
                new_allocation = dict(allocation)
                new_allocation[from_mixer] -= 1
                new_allocation[to_mixer] += 1
                yield new_allocation, 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_mixers(self) -> int:
        return len(self.config.mixer_names)

    @property
    def phase_order(self) -> list[str]:
        return list(self.config.mixer_names)


class UniformCostModelConfig(BaseCostModelConfig):
    """Configuration for the uniform (zero-cost) model.

    The only required field beyond the type tag is ``mixer_names``,
    which lists the mixer types in phase order.
    """

    _impl_class: ClassVar[type[BaseCostModel]] = UniformCostModel

    cost_model_type: Literal["uniform"] = "uniform"

    mixer_names: list[str] = Field(
        description="Mixer type names, in phase order.",
    )

    @model_validator(mode="after")
    def validate_mixer_names(self) -> Self:
        if len(self.mixer_names) < 2:
            raise ValueError("Must have at least 2 mixer types")
        if len(self.mixer_names) != len(set(self.mixer_names)):
            raise ValueError("Mixer names must be unique")
        return self
