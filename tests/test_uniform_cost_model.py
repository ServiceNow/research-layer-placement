"""Tests for UniformCostModel."""

from math import comb

import pytest

from place_layers.core.types import LayerPlacement
from place_layers.cost_model.uniform_cost_model import (
    UniformCostModel,
    UniformCostModelConfig,
)


# =============================================================================
# Fixtures
# =============================================================================


def make_uniform_model(mixer_names: list[str]) -> UniformCostModel:
    config = UniformCostModelConfig(
        cost_model_type="uniform",
        mixer_names=mixer_names,
    )
    return config.create_cost_model()


@pytest.fixture
def two_mixer() -> UniformCostModel:
    return make_uniform_model(["m0", "m1"])


@pytest.fixture
def three_mixer() -> UniformCostModel:
    return make_uniform_model(["m0", "m1", "m2"])


# =============================================================================
# Config validation
# =============================================================================


class TestUniformCostModelConfig:
    def test_valid_config(self) -> None:
        config = UniformCostModelConfig(
            cost_model_type="uniform",
            mixer_names=["a", "b"],
        )
        assert config.mixer_names == ["a", "b"]

    def test_too_few_mixers(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            UniformCostModelConfig(
                cost_model_type="uniform",
                mixer_names=["a"],
            )

    def test_duplicate_mixers(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            UniformCostModelConfig(
                cost_model_type="uniform",
                mixer_names=["a", "a"],
            )

    def test_smart_constructor(self) -> None:
        config = UniformCostModelConfig(
            cost_model_type="uniform",
            mixer_names=["a", "b", "c"],
        )
        model = config.create_cost_model()
        assert isinstance(model, UniformCostModel)
        assert model.num_mixers == 3


# =============================================================================
# Cost operations
# =============================================================================


class TestCostOperations:
    def test_cost_is_zero(self, two_mixer: UniformCostModel) -> None:
        placement = LayerPlacement(assignments=("m0", "m1", "m0", "m1"))
        assert two_mixer.cost(placement) == 0.0

    def test_cost_delta_is_zero(self, two_mixer: UniformCostModel) -> None:
        assert two_mixer.cost_delta(0, "m0", "m1") == 0.0
        assert two_mixer.cost_delta(5, "m1", "m0") == 0.0

    def test_min_cost_is_zero(self, two_mixer: UniformCostModel) -> None:
        assert two_mixer.min_cost(10) == 0.0

    def test_max_cost_is_zero(self, two_mixer: UniformCostModel) -> None:
        assert two_mixer.max_cost(10) == 0.0

    def test_allocation_cost_is_zero(self, two_mixer: UniformCostModel) -> None:
        assert two_mixer.allocation_cost({"m0": 3, "m1": 7}) == 0.0


# =============================================================================
# Allocation operations
# =============================================================================


class TestAllocationOperations:
    def test_min_cost_allocation(self, three_mixer: UniformCostModel) -> None:
        alloc = three_mixer.min_cost_allocation(10)
        assert sum(alloc.values()) == 10
        assert set(alloc.keys()) == {"m0", "m1", "m2"}

    def test_max_cost_allocation(self, three_mixer: UniformCostModel) -> None:
        alloc = three_mixer.max_cost_allocation(10)
        assert sum(alloc.values()) == 10
        assert set(alloc.keys()) == {"m0", "m1", "m2"}

    def test_allocation_at_cost_bound_nonneg(self, two_mixer: UniformCostModel) -> None:
        alloc = two_mixer.allocation_at_cost_bound(6, cost_bound=0.0)
        assert alloc is not None
        assert sum(alloc.values()) == 6

    def test_allocation_at_cost_bound_negative(self, two_mixer: UniformCostModel) -> None:
        alloc = two_mixer.allocation_at_cost_bound(6, cost_bound=-1.0)
        assert alloc is None

    def test_allocation_neighbors_all_zero_delta(
        self, three_mixer: UniformCostModel
    ) -> None:
        alloc = {"m0": 3, "m1": 4, "m2": 3}
        neighbors = list(three_mixer.allocation_neighbors(alloc, "increasing"))
        for _, delta in neighbors:
            assert delta == 0.0
        # K*(K-1) moves when all counts > 0 (excluding zero-count sources)
        assert len(neighbors) == 3 * 2  # 3 sources * 2 targets each


# =============================================================================
# allocations_by_cost — exhaustive enumeration
# =============================================================================


class TestAllocationsByCost:
    @pytest.mark.parametrize(
        "num_layers,num_mixers",
        [(4, 2), (6, 3), (5, 4), (8, 2), (3, 3)],
    )
    def test_count_matches_stars_and_bars(
        self, num_layers: int, num_mixers: int
    ) -> None:
        """allocations_by_cost should yield C(N+K-1, K-1) allocations."""
        mixer_names = [f"m{i}" for i in range(num_mixers)]
        model = make_uniform_model(mixer_names)
        expected = comb(num_layers + num_mixers - 1, num_mixers - 1)

        allocs = list(model.allocations_by_cost(num_layers, direction="increasing"))
        assert len(allocs) == expected

    @pytest.mark.parametrize(
        "num_layers,num_mixers",
        [(4, 2), (6, 3), (5, 4)],
    )
    def test_all_costs_zero(self, num_layers: int, num_mixers: int) -> None:
        """Every yielded allocation should have cost 0."""
        mixer_names = [f"m{i}" for i in range(num_mixers)]
        model = make_uniform_model(mixer_names)

        for cost, _alloc in model.allocations_by_cost(num_layers):
            assert cost == 0.0

    def test_no_duplicates(self, three_mixer: UniformCostModel) -> None:
        """No duplicate allocations should be yielded."""
        allocs = list(three_mixer.allocations_by_cost(6))
        keys = [frozenset(a.items()) for _, a in allocs]
        assert len(keys) == len(set(keys))

    def test_decreasing_direction(self, two_mixer: UniformCostModel) -> None:
        """Decreasing direction should yield the same set of allocations."""
        n = 5
        inc = list(two_mixer.allocations_by_cost(n, direction="increasing"))
        dec = list(two_mixer.allocations_by_cost(n, direction="decreasing"))
        assert len(inc) == len(dec)

    def test_phase_order(self, three_mixer: UniformCostModel) -> None:
        """phase_order should match the configured mixer_names."""
        assert three_mixer.phase_order == ["m0", "m1", "m2"]

    def test_num_mixers(self, three_mixer: UniformCostModel) -> None:
        assert three_mixer.num_mixers == 3


# =============================================================================
# Discriminated union
# =============================================================================


class TestDiscriminatedUnion:
    def test_uniform_in_union(self) -> None:
        """UniformCostModelConfig should be selectable via discriminator."""
        from pydantic import TypeAdapter

        from place_layers.cost_model import CostModelConfig

        adapter = TypeAdapter(CostModelConfig)
        config = adapter.validate_python(
            {"cost_model_type": "uniform", "mixer_names": ["a", "b"]}
        )
        assert isinstance(config, UniformCostModelConfig)

    def test_linear_still_works(self) -> None:
        """LinearCostModelConfig should still be selectable."""
        from pydantic import TypeAdapter

        from place_layers.cost_model import CostModelConfig

        adapter = TypeAdapter(CostModelConfig)
        config = adapter.validate_python(
            {"cost_model_type": "linear", "mixer_costs": {"a": 1.0, "b": 0.5}}
        )
        from place_layers.cost_model.linear_cost_model import LinearCostModelConfig

        assert isinstance(config, LinearCostModelConfig)
