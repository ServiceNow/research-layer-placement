"""Chain MRF model for layer placement optimization.

This module separates the MRF model from the evaluator, enabling:
- Exact optimization via DP (Viterbi algorithm)
- Spectral analysis of the transfer matrix
- True score computation without noise
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from place_layers.core.types import LayerPlacement


@dataclass(frozen=True)
class ChainMRFModel:
    """Represents a chain MRF with unary and pairwise potentials.

    The score function is:
        Score(x) = sum_i phi_i(x_i) + sum_i psi_i(x_i, x_{i+1})

    where:
        - phi_i (unary potential): Reward/penalty based on layer_needs[i]
        - psi_i (pairwise potential): Interaction between adjacent layers

    This model represents the "true" score without noise.
    """

    mixer_names: tuple[str, ...]
    layer_needs: tuple[str, ...]  # Preferred mixer name per layer
    pairwise_weights: tuple[float, ...]  # Interaction weights (length = num_layers - 1)
    unary_bonus: float
    unary_penalty: float
    pairwise_scale: float

    @property
    def num_layers(self) -> int:
        return len(self.layer_needs)

    @property
    def num_mixers(self) -> int:
        return len(self.mixer_names)

    def __post_init__(self) -> None:
        """Validate model parameters."""
        if len(self.pairwise_weights) != self.num_layers - 1:
            raise ValueError(
                f"pairwise_weights length ({len(self.pairwise_weights)}) "
                f"must equal num_layers - 1 ({self.num_layers - 1})"
            )
        valid_names = set(self.mixer_names)
        for i, name in enumerate(self.layer_needs):
            if name not in valid_names:
                raise ValueError(
                    f"layer_needs[{i}] = {name!r} is not in mixer_names {self.mixer_names}"
                )

    @classmethod
    def from_seed(
        cls,
        *,
        seed: int,
        num_layers: int,
        mixer_names: Sequence[str],
        unary_bonus: float,
        unary_penalty: float,
        pairwise_scale: float,
    ) -> ChainMRFModel:
        """Generate an MRF landscape from a random seed.

        Args:
            seed: Random seed for reproducible generation.
            num_layers: Number of layers.
            mixer_names: Available mixer names.
            unary_bonus: Score bonus when mixer matches layer's need.
            unary_penalty: Score penalty when mixer doesn't match.
            pairwise_scale: Scale factor for pairwise interaction terms.

        Returns:
            ChainMRFModel with randomly generated landscape.
        """
        rng = random.Random(seed)
        names = tuple(mixer_names)
        layer_needs = tuple(names[rng.randint(0, len(names) - 1)] for _ in range(num_layers))
        pairwise_weights = tuple(rng.gauss(0.0, 1.0) for _ in range(num_layers - 1))
        return cls(
            mixer_names=names,
            layer_needs=layer_needs,
            pairwise_weights=pairwise_weights,
            unary_bonus=unary_bonus,
            unary_penalty=unary_penalty,
            pairwise_scale=pairwise_scale,
        )

    def unary_potential(self, layer: int, mixer_name: str) -> float:
        """Compute unary potential phi_i(k).

        Args:
            layer: Layer index (0 to num_layers - 1).
            mixer_name: Mixer name.

        Returns:
            +unary_bonus if mixer matches layer's need, else -unary_penalty.
        """
        if mixer_name == self.layer_needs[layer]:
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
        weight = self.pairwise_weights[layer]
        similarity = 1.0 if mixer_a == mixer_b else -1.0
        return weight * similarity * self.pairwise_scale

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

        # Pairwise potentials
        for i in range(len(placement.assignments) - 1):
            total_score += self.pairwise_potential(
                i, placement.assignments[i], placement.assignments[i + 1]
            )

        return total_score
