"""Chain MRF model for layer placement optimization.

This module separates the MRF model from the evaluator, enabling:
- Exact optimization via DP (Viterbi algorithm)
- Spectral analysis of the transfer matrix
- True score computation without noise
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from place_layers.core.types import LayerPlacement


@runtime_checkable
class MRFModelProtocol(Protocol):
    """Protocol capturing the interface used by ExactMRFSolver."""

    @property
    def num_layers(self) -> int: ...

    @property
    def num_mixers(self) -> int: ...

    @property
    def mixer_names(self) -> tuple[str, ...]: ...

    @property
    def memory(self) -> int: ...

    @property
    def pairwise_max_range(self) -> int: ...

    @property
    def triplet_scale(self) -> float: ...

    def unary_potential(self, layer: int, mixer_name: str) -> float: ...

    def pairwise_potential(self, layer: int, mixer_a: str, mixer_b: str) -> float: ...

    def skip_pairwise_potential(
        self, layer: int, range_r: int, mixer_a: str, mixer_b: str
    ) -> float: ...

    def triplet_potential(
        self, layer: int, mixer_a: str, mixer_b: str, mixer_c: str
    ) -> float: ...


@dataclass(frozen=True)
class ChainMRFModel:
    """Represents a chain MRF with unary and pairwise potentials.

    The score function is:
        Score(x) = sum_i phi_i(x_i) + sum_i psi_i(x_i, x_{i+1})
                 + sum_{r>=2} sum_i skip_psi_{i,r}(x_i, x_{i+r})
                 + sum_i triplet(x_i, x_{i+1}, x_{i+2})

    where:
        - phi_i (unary potential): Reward/penalty based on layer_needs[i]
        - psi_i (pairwise potential): Interaction between adjacent layers
        - skip_psi_{i,r}: Skip-pairwise interaction at range r
        - triplet: 3-body interaction for consecutive triples

    All random structure is generated internally from the seed.
    Scalar knobs (pairwise_decay, pairwise_max_range, triplet_scale)
    control the ladder from localized to delocalized, low to high order.
    """

    mixer_names: tuple[str, ...]
    seed: int
    num_layers: int
    unary_bonus: float
    unary_penalty: float
    pairwise_scale: float
    pairwise_decay: float = 0.5     # range-r scale = pairwise_scale * decay^(r-1)
    pairwise_max_range: int = 1     # 1 = adjacent only (baseline)
    triplet_scale: float = 0.0      # 0.0 = disabled

    @property
    def layer_needs(self) -> tuple[str, ...]:
        return self._layer_needs  # type: ignore[attr-defined]

    @property
    def pairwise_weights(self) -> tuple[float, ...]:
        return self._pairwise_weights  # type: ignore[attr-defined]

    @property
    def num_mixers(self) -> int:
        return len(self.mixer_names)

    @property
    def memory(self) -> int:
        """Number of past layers the DP must track."""
        m = 1
        if self.pairwise_max_range > 1:
            m = max(m, self.pairwise_max_range)
        if self.triplet_scale > 0:
            m = max(m, 2)
        return m

    def __post_init__(self) -> None:
        """Validate parameters and generate all random structure."""
        # Normalize mixer_names to tuple (accepts list from Pydantic configs)
        if not isinstance(self.mixer_names, tuple):
            object.__setattr__(self, "mixer_names", tuple(self.mixer_names))

        # Validate
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {self.num_layers}")
        if self.pairwise_max_range < 1:
            raise ValueError(
                f"pairwise_max_range must be >= 1, got {self.pairwise_max_range}"
            )
        if self.pairwise_decay < 0:
            raise ValueError(f"pairwise_decay must be >= 0, got {self.pairwise_decay}")
        if len(self.mixer_names) == 0:
            raise ValueError("mixer_names must not be empty")

        N = self.num_layers
        K = self.num_mixers
        names = self.mixer_names
        rng = random.Random(self.seed)

        # --- RNG sequence (order preserved for seed-identical backward compat) ---
        layer_needs = tuple(names[rng.randint(0, K - 1)] for _ in range(N))
        pairwise_weights = tuple(rng.gauss(0.0, 1.0) for _ in range(N - 1))

        object.__setattr__(self, "_layer_needs", layer_needs)
        object.__setattr__(self, "_pairwise_weights", pairwise_weights)

        # Skip-pairwise potentials (skipped entirely when pairwise_max_range == 1)
        skip_potentials: dict[int, tuple[tuple[tuple[float, ...], ...], ...]] = {}
        for r in range(2, self.pairwise_max_range + 1):
            scale = self.pairwise_scale * self.pairwise_decay ** (r - 1)
            matrices = tuple(
                tuple(
                    tuple(rng.gauss(0.0, scale) for _c in range(K))
                    for _b in range(K)
                )
                for _ in range(N - r)
            )
            skip_potentials[r] = matrices
        object.__setattr__(self, "_skip_potentials", skip_potentials)

        # Triplet potentials (skipped entirely when triplet_scale == 0)
        triplet_potentials = None
        if self.triplet_scale > 0.0:
            triplet_potentials = tuple(
                tuple(
                    tuple(
                        tuple(
                            rng.gauss(0.0, self.triplet_scale) for _d in range(K)
                        )
                        for _c in range(K)
                    )
                    for _b in range(K)
                )
                for _ in range(N - 2)
            )
        object.__setattr__(self, "_triplet_potentials", triplet_potentials)

    def unary_potential(self, layer: int, mixer_name: str) -> float:
        """Compute unary potential phi_i(k).

        Args:
            layer: Layer index (0 to num_layers - 1).
            mixer_name: Mixer name.

        Returns:
            +unary_bonus if mixer matches layer's need, else -unary_penalty.
        """
        if mixer_name == self._layer_needs[layer]:  # type: ignore[attr-defined]
            return self.unary_bonus
        return -self.unary_penalty

    def pairwise_potential(self, layer: int, mixer_a: str, mixer_b: str) -> float:
        """Compute pairwise potential psi_i(a, b).

        This is the interaction term between layer i (with mixer_a) and
        layer i+1 (with mixer_b).

        Args:
            layer: Layer index i (0 to num_layers - 2).
            mixer_a: Mixer name for layer i.
            mixer_b: Mixer name for layer i+1.

        Returns:
            weight * similarity * pairwise_scale
        """
        weight = self._pairwise_weights[layer]  # type: ignore[attr-defined]
        similarity = 1.0 if mixer_a == mixer_b else -1.0
        return weight * similarity * self.pairwise_scale

    def skip_pairwise_potential(
        self, layer: int, range_r: int, mixer_a: str, mixer_b: str
    ) -> float:
        """Compute skip-pairwise potential for layers at distance r.

        Args:
            layer: Layer index i (0 to num_layers - range_r - 1).
            range_r: Distance between layers (>= 2).
            mixer_a: Mixer name for layer i.
            mixer_b: Mixer name for layer i + range_r.

        Returns:
            Potential value from the lookup table.
        """
        a = self.mixer_names.index(mixer_a)
        b = self.mixer_names.index(mixer_b)
        return self._skip_potentials[range_r][layer][a][b]  # type: ignore[attr-defined]

    def triplet_potential(
        self, layer: int, mixer_a: str, mixer_b: str, mixer_c: str
    ) -> float:
        """Compute triplet (3-body) potential for layers (i, i+1, i+2).

        Args:
            layer: Layer index i (0 to num_layers - 3).
            mixer_a: Mixer name for layer i.
            mixer_b: Mixer name for layer i+1.
            mixer_c: Mixer name for layer i+2.

        Returns:
            Potential value from the lookup table.
        """
        assert self._triplet_potentials is not None  # type: ignore[attr-defined]
        a = self.mixer_names.index(mixer_a)
        b = self.mixer_names.index(mixer_b)
        c = self.mixer_names.index(mixer_c)
        return self._triplet_potentials[layer][a][b][c]  # type: ignore[attr-defined]

    def true_score(self, placement: LayerPlacement) -> float:
        """Compute MRF score WITHOUT noise.

        This is the deterministic part of the score function, used for
        computing regret (difference between optimal and achieved score).

        Args:
            placement: Layer-to-mixer assignment.

        Returns:
            True score (higher is better).
        """
        total_score = 0.0

        # Unary potentials
        for layer_idx, mixer_name in enumerate(placement.assignments):
            total_score += self.unary_potential(layer_idx, mixer_name)

        # Pairwise potentials (range 1)
        for i in range(len(placement.assignments) - 1):
            total_score += self.pairwise_potential(
                i, placement.assignments[i], placement.assignments[i + 1]
            )

        # Skip-pairwise potentials (range >= 2)
        for r in range(2, self.pairwise_max_range + 1):
            for i in range(self.num_layers - r):
                total_score += self.skip_pairwise_potential(
                    i, r, placement.assignments[i], placement.assignments[i + r]
                )

        # Triplet potentials
        if self.triplet_scale > 0:
            for i in range(self.num_layers - 2):
                total_score += self.triplet_potential(
                    i,
                    placement.assignments[i],
                    placement.assignments[i + 1],
                    placement.assignments[i + 2],
                )

        return total_score
