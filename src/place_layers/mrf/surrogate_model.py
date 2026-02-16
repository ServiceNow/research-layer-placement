"""SurrogateMRFModel — wraps ridge coefficients as MRF potentials.

The ridge prediction `coef @ encode(x) + intercept` is literally a chain MRF
score.  Each coefficient IS the MRF potential for its corresponding feature.
This class extracts those coefficients into the MRFModelProtocol interface so
the ExactMRFSolver can find exact optimal placements via Viterbi DP.
"""

from __future__ import annotations

import numpy as np

from place_layers.surrogate.encoding import FeatureExpansionConfig


class SurrogateMRFModel:
    """MRF model backed by ridge regression coefficients.

    Implements MRFModelProtocol.  Constructor walks the coefficient vector
    (same layout as encoding.py) and reshapes into unary, pairwise,
    skip-pairwise, triplet, and count-bias arrays.

    Count bias is folded into unary potentials so that the DP score equals
    ``ridge_prediction - intercept`` exactly (intercept added back at end).
    """

    def __init__(
        self,
        coef: np.ndarray,
        intercept: float,
        num_layers: int,
        mixer_names: tuple[str, ...],
        feature_config: FeatureExpansionConfig,
    ) -> None:
        assert mixer_names == tuple(sorted(mixer_names)), (
            f"mixer_names must be sorted, got {mixer_names}"
        )
        assert feature_config.max_order <= 3, (
            f"Solver doesn't handle order > 3, got {feature_config.max_order}"
        )

        self._num_layers = num_layers
        self._mixer_names = mixer_names
        self._intercept = intercept
        self._feature_config = feature_config

        N = num_layers
        K = len(mixer_names)
        self._name_to_idx = {name: i for i, name in enumerate(mixer_names)}

        offset = 0

        # --- Unary: N*K coefficients reshaped to (N, K) ---
        self._unary = coef[offset : offset + N * K].reshape(N, K).copy()
        offset += N * K

        # --- Pairwise indicators ---
        self._pairwise: dict[int, np.ndarray] = {}
        if feature_config.max_order >= 2:
            K2 = K * K
            for r in range(1, feature_config.pairwise_max_range + 1):
                n_pairs = max(0, N - r)
                block = coef[offset : offset + n_pairs * K2].reshape(n_pairs, K, K)
                self._pairwise[r] = block.copy()
                offset += n_pairs * K2

        # --- Triplet indicators (order 3) ---
        self._triplets: np.ndarray | None = None
        if feature_config.max_order >= 3:
            K3 = K ** 3
            n_triples = max(0, N - 2)
            self._triplets = coef[offset : offset + n_triples * K3].reshape(
                n_triples, K, K, K
            ).copy()
            offset += n_triples * K3

        # --- Count bias: fold into unary ---
        if feature_config.allocation_counts:
            count_coef = coef[offset : offset + K].copy()
            offset += K
            # effective_unary[i, j] += count_coef[j]
            self._unary += count_coef[np.newaxis, :]

        assert offset == len(coef), (
            f"Coefficient vector length mismatch: consumed {offset}, got {len(coef)}"
        )

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_mixers(self) -> int:
        return len(self._mixer_names)

    @property
    def mixer_names(self) -> tuple[str, ...]:
        return self._mixer_names

    @property
    def intercept(self) -> float:
        return self._intercept

    @property
    def pairwise_max_range(self) -> int:
        if self._feature_config.max_order < 2:
            return 1
        return self._feature_config.pairwise_max_range

    @property
    def triplet_scale(self) -> float:
        return 1.0 if self._triplets is not None else 0.0

    @property
    def memory(self) -> int:
        m = 1
        if self.pairwise_max_range > 1:
            m = max(m, self.pairwise_max_range)
        if self._triplets is not None:
            m = max(m, 2)
        return m

    def unary_potential(self, layer: int, mixer_name: str) -> float:
        return float(self._unary[layer, self._name_to_idx[mixer_name]])

    def pairwise_potential(self, layer: int, mixer_a: str, mixer_b: str) -> float:
        if 1 not in self._pairwise:
            return 0.0
        a = self._name_to_idx[mixer_a]
        b = self._name_to_idx[mixer_b]
        return float(self._pairwise[1][layer, a, b])

    def skip_pairwise_potential(
        self, layer: int, range_r: int, mixer_a: str, mixer_b: str
    ) -> float:
        if range_r not in self._pairwise:
            return 0.0
        a = self._name_to_idx[mixer_a]
        b = self._name_to_idx[mixer_b]
        return float(self._pairwise[range_r][layer, a, b])

    def triplet_potential(
        self, layer: int, mixer_a: str, mixer_b: str, mixer_c: str
    ) -> float:
        if self._triplets is None:
            return 0.0
        a = self._name_to_idx[mixer_a]
        b = self._name_to_idx[mixer_b]
        c = self._name_to_idx[mixer_c]
        return float(self._triplets[layer, a, b, c])
