"""Mock supernet evaluator for layer placement optimization."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from typing import ClassVar, Literal

from pydantic import Field

from place_layers.core.types import LayerPlacement
from place_layers.evaluator.base_evaluator import BaseEvaluator, BaseEvaluatorConfig
from place_layers.mrf.model import ChainMRFModel


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

    config: "MockSupernetEvaluatorConfig"

    def __init__(self, config: "MockSupernetEvaluatorConfig") -> None:
        super().__init__(config)

        self._mrf_model = ChainMRFModel.from_seed(
            seed=self.config.seed,
            num_layers=self.config.num_layers,
            mixer_names=self.config.mixer_names,
            unary_bonus=self.config.unary_bonus,
            unary_penalty=self.config.unary_penalty,
            pairwise_scale=self.config.pairwise_scale,
        )

    @classmethod
    def with_noise(
        cls,
        evaluator: MockSupernetEvaluator,
        noise_sigma: float,
        noise_seed_shift: int = 0,
    ) -> MockSupernetEvaluator:
        """Return a copy of *evaluator* with a different noise configuration."""
        config = evaluator.config.model_copy(
            update={"noise_sigma": noise_sigma, "noise_seed_shift": noise_seed_shift}
        )
        return cls(config)

    @property
    def mrf_model(self) -> ChainMRFModel:
        """The underlying MRF model defining the score landscape."""
        return self._mrf_model

    @property
    def num_layers(self) -> int:
        """Number of layers (from config, since this is a mock evaluator)."""
        return self.config.num_layers

    @property
    def mixer_names(self) -> frozenset[str]:
        """Mixer names from config.

        This is for validation only. For ordering, use cost_model.phase_order.
        """
        return frozenset(self.config.mixer_names)

    async def evaluate(
        self, parameters: LayerPlacement, *, noise_sigma: float | None = None
    ) -> float:
        """
        Evaluate parameters using the MRF energy model.

        Score = Σᵢ unary(i) + Σᵢ pairwise(i, i+1) + noise

        Args:
            parameters: Parameters to evaluate (assignments are mixer names).
            noise_sigma: Override noise level. None uses config default, 0.0 means no noise.

        Returns:
            Score (higher is better).
        """
        effective_sigma = (
            noise_sigma if noise_sigma is not None else self.config.noise_sigma
        )
        total_score = self._mrf_model.true_score(parameters)

        if effective_sigma > 0:
            # DETERMINISTIC NOISE: Parameter-specific perturbation
            # noise_seed_shift produces independent noise realizations
            # over the same landscape
            placement_hash = int(
                hashlib.sha256(
                    json.dumps(parameters.assignments).encode()
                ).hexdigest()[:16],
                16,
            )
            noise_seed = (
                placement_hash + self.config.noise_seed_shift
            ) % (2**64)
            noise_rng = random.Random(noise_seed)
            total_score += noise_rng.gauss(0.0, effective_sigma)

        # Optional delay for testing display dynamics
        if self.config.eval_delay_seconds > 0:
            await asyncio.sleep(self.config.eval_delay_seconds)

        return total_score


class MockSupernetEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for mock supernet evaluator with MRF landscape."""

    _impl_class: ClassVar[type[BaseEvaluator]] = MockSupernetEvaluator

    evaluator_type: Literal["mock_supernet"] = "mock_supernet"

    num_layers: int = Field(
        description="Number of layers (must match problem config)", gt=0
    )

    mixer_names: list[str] = Field(
        description="Mixer names (e.g., ['FA', 'SWA', 'GDN', 'KDA'])",
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

    noise_seed_shift: int = Field(
        default=0,
        description="Additive shift applied to the placement hash before seeding "
        "the noise RNG. Different shifts produce independent noise realizations "
        "over the same MRF landscape.",
    )

    eval_delay_seconds: float = Field(
        description="Artificial delay per evaluation (seconds) for testing display dynamics",
        ge=0.0,
    )
