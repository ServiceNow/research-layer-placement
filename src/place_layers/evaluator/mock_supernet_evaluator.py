"""Mock supernet evaluator for layer placement optimization."""

import hashlib
import json
import random
from typing import ClassVar, Literal

from pydantic import Field

from place_layers.core.types import LayerPlacement
from place_layers.evaluator.base_evaluator import BaseEvaluator
from place_layers.evaluator.base_evaluator_config import BaseEvaluatorConfig


class MockSupernetEvaluator(BaseEvaluator[LayerPlacement]):
    """
    Mock evaluator simulating a trained neural network with multiple layer-wise options.

    CONTEXT:
    Consider a neural network with N layers, where each layer can use one of M
    different implementations (e.g., full attention, sliding window attention,
    linear attention). A "supernet" is such a network trained with stochastic
    sampling - at each training step, a random implementation is active at each
    layer. After training, we want to find the best configuration: which
    implementation to use at each layer, subject to budget constraints on how
    many times each implementation can be used.

    MODEL:
    This evaluator simulates the performance landscape of such a supernet using
    a Markov Random Field (MRF):

    1. LATENT STRUCTURE (generated once from seed):
       - layer_needs[i]: Which implementation works best at layer i
       - pairwise_weights[i]: How adjacent layers interact

    2. SCORE FUNCTION (chain MRF over configurations):
       Score(config) = Σᵢ φᵢ(config[i]) + Σᵢ ψᵢ(config[i], config[i+1]) + noise

       where:
       - φᵢ (unary potential): Reward/penalty based on layer_needs[i]
       - ψᵢ (pairwise potential): Interaction between adjacent layers
       - noise: Parameter-specific Gaussian perturbation

    3. KEY PROPERTIES:
       - Budget-agnostic: Landscape exists independently of optimization constraints
       - Deterministic: Same configuration always receives the same score
       - Non-trivial: Local structure makes search tractable but not trivial
       - Realistic: Models arbitrary learned preferences from stochastic training

    The optimization task is: find the highest-scoring configuration subject to
    budget constraints (e.g., use at most n full attention layers, m sliding
    window layers, etc.).
    """

    def __init__(self, config: "MockSupernetEvaluatorConfig") -> None:
        super().__init__(config)
        self.config: MockSupernetEvaluatorConfig  # Type narrowing

        # Generate landscape structure from seed
        # This represents the supernet's learned preferences
        rng = random.Random(self.config.seed)

        # Unary potentials: Each layer's preferred implementation
        self.layer_needs = [
            rng.randint(0, self.config.num_mixers - 1)
            for _ in range(self.config.num_layers)
        ]

        # Pairwise potentials: Interaction strength between adjacent layers
        # Positive weight: adjacent layers benefit from same implementation
        # Negative weight: adjacent layers benefit from different implementations
        self.pairwise_weights = [
            rng.gauss(0.0, 1.0) for _ in range(self.config.num_layers - 1)
        ]

    @property
    def num_layers(self) -> int:
        """Number of layers (from config, since this is a mock evaluator)."""
        return self.config.num_layers

    @property
    def num_mixers(self) -> int:
        """Number of mixer types (from config, since this is a mock evaluator)."""
        return self.config.num_mixers

    async def evaluate(self, parameters: LayerPlacement) -> float:
        """
        Evaluate parameters using the MRF energy model.

        Score = Σᵢ unary(i) + Σᵢ pairwise(i, i+1) + noise

        Args:
            parameters: Parameters to evaluate.

        Returns:
            Score (higher is better).
        """
        total_score = 0.0

        # UNARY POTENTIALS: Does mixer match layer's intrinsic need?
        for layer_idx, mixer_idx in enumerate(parameters.assignments):
            if mixer_idx == self.layer_needs[layer_idx]:
                # Correct mixer for this layer
                total_score += self.config.unary_bonus
            else:
                # Wrong mixer for this layer
                total_score -= self.config.unary_penalty

        # PAIRWISE POTENTIALS: Do adjacent layers work well together?
        for i in range(len(parameters.assignments) - 1):
            mixer_i = parameters.assignments[i]
            mixer_j = parameters.assignments[i + 1]

            # Interaction weight (can be positive or negative)
            weight = self.pairwise_weights[i]

            # Similarity: 1 if same mixer, -1 if different
            similarity = 1.0 if mixer_i == mixer_j else -1.0

            # Contribution: weight * similarity * scale
            total_score += weight * similarity * self.config.pairwise_scale

        # DETERMINISTIC NOISE: Parameter-specific perturbation
        params_hash = hashlib.sha256(
            json.dumps(parameters.assignments).encode()
        ).hexdigest()
        noise_seed = self.config.seed ^ int(params_hash[:16], 16) & 0xFFFFFFFF
        noise_rng = random.Random(noise_seed)
        total_score += noise_rng.gauss(0.0, self.config.noise_sigma)

        return total_score


class MockSupernetEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for mock supernet evaluator with MRF landscape."""

    _impl_class: ClassVar[type[BaseEvaluator]] = MockSupernetEvaluator

    evaluator_type: Literal["mock_supernet"] = "mock_supernet"

    num_layers: int = Field(
        description="Number of layers (must match problem config)", gt=0
    )

    num_mixers: int = Field(
        description="Number of mixers (must match problem config)", gt=1
    )

    seed: int = Field(
        description="Random seed for generating the landscape structure. "
        "Represents the supernet's learned preferences from training.",
    )

    unary_bonus: float = Field(
        description="Score bonus when mixer matches layer's intrinsic need",
        gt=0.0,
    )

    unary_penalty: float = Field(
        description="Score penalty when mixer doesn't match layer's need",
        ge=0.0,
    )

    pairwise_scale: float = Field(
        description="Scale factor for pairwise interaction terms. "
        "Controls how much adjacent layers affect each other.",
        ge=0.0,
    )

    noise_sigma: float = Field(
        description="Standard deviation of configuration-specific Gaussian noise",
        ge=0.0,
    )
