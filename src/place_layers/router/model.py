"""Torch reward-model utilities for prompt-conditioned placement scoring."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Annotated, Literal

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch import nn

from place_layers.router.metrics import RewardModelMetrics
from place_layers.surrogate.encoding import (
    FeatureExpansionConfig,
    build_type_mapping,
    compute_feature_dim,
    encode_batch,
)


class RewardModelFitResult(BaseModel):
    """Summary artifact from fit_reward_model."""

    model_type: str
    optimizer_type: str
    training_objective: str = "mse_loss"
    teacher_temperature: float | None = None
    student_temperature: float | None = None
    num_epochs: int
    batch_size: int
    num_train_examples: int
    num_test_examples: int
    num_train_prompts: int
    num_test_prompts: int
    num_placements: int
    precision_k: int
    embedding_dim: int
    placement_feature_dim: int
    feature_config: FeatureExpansionConfig
    target_mean: float | None
    target_std: float | None
    train_metrics: RewardModelMetrics
    test_metrics: RewardModelMetrics


class BaseRewardModelConfig(BaseModel, ABC):
    """Base configuration for reward-model architectures."""

    model_type: str = Field(description="Reward-model architecture discriminator.")
    hidden_dims: list[int] = Field(
        default_factory=lambda: [256, 128],
        description="Hidden layer sizes for the prompt encoder MLP.",
    )
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Dropout probability applied between reward head layers.",
    )
    activation: Literal["relu", "gelu", "tanh"] = Field(
        default="gelu",
        description="Activation function used in the reward head.",
    )

    @abstractmethod
    def create_model(
        self,
        *,
        embedding_dim: int,
        placement_feature_dim: int,
    ) -> nn.Module:
        """Construct the configured reward model."""


class MeanPoolingRewardModelConfig(BaseRewardModelConfig):
    """Reward model that mean-pools prompt token embeddings."""

    model_type: Literal["mean_pool"] = "mean_pool"

    def create_model(
        self,
        *,
        embedding_dim: int,
        placement_feature_dim: int,
    ) -> nn.Module:
        return MeanPoolingRewardModel(
            embedding_dim=embedding_dim,
            placement_feature_dim=placement_feature_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            activation=self.activation,
        )


class AttentionPoolingRewardModelConfig(BaseRewardModelConfig):
    """Reward model with learned attention pooling over prompt tokens."""

    model_type: Literal["attention_pool"] = "attention_pool"
    attention_hidden_dim: int = Field(
        default=128,
        gt=0,
        description="Hidden size for the token-attention scorer.",
    )

    def create_model(
        self,
        *,
        embedding_dim: int,
        placement_feature_dim: int,
    ) -> nn.Module:
        return AttentionPoolingRewardModel(
            embedding_dim=embedding_dim,
            placement_feature_dim=placement_feature_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            activation=self.activation,
            attention_hidden_dim=self.attention_hidden_dim,
        )


RewardModelConfig = Annotated[
    MeanPoolingRewardModelConfig | AttentionPoolingRewardModelConfig,
    Field(discriminator="model_type"),
]


class BaseOptimizerConfig(BaseModel, ABC):
    """Base configuration for reward-model optimizers."""

    optimizer_type: str = Field(description="Optimizer discriminator.")
    learning_rate: float = Field(gt=0.0, description="Optimizer learning rate.")
    weight_decay: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight decay applied by the optimizer.",
    )

    @abstractmethod
    def create_optimizer(self, parameters: Sequence[torch.nn.Parameter]) -> torch.optim.Optimizer:
        """Construct the configured optimizer."""


class AdamOptimizerConfig(BaseOptimizerConfig):
    """Adam optimizer configuration."""

    optimizer_type: Literal["adam"] = "adam"
    learning_rate: float = Field(default=1e-3, gt=0.0)

    def create_optimizer(self, parameters: Sequence[torch.nn.Parameter]) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


class AdamWOptimizerConfig(BaseOptimizerConfig):
    """AdamW optimizer configuration."""

    optimizer_type: Literal["adamw"] = "adamw"
    learning_rate: float = Field(default=1e-3, gt=0.0)

    def create_optimizer(self, parameters: Sequence[torch.nn.Parameter]) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


class SGDOptimizerConfig(BaseOptimizerConfig):
    """SGD optimizer configuration."""

    optimizer_type: Literal["sgd"] = "sgd"
    learning_rate: float = Field(default=1e-2, gt=0.0)
    momentum: float = Field(default=0.0, ge=0.0, lt=1.0)
    nesterov: bool = Field(default=False)

    def create_optimizer(self, parameters: Sequence[torch.nn.Parameter]) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            parameters,
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
        )


RewardOptimizerConfig = Annotated[
    AdamOptimizerConfig | AdamWOptimizerConfig | SGDOptimizerConfig,
    Field(discriminator="optimizer_type"),
]


def build_placement_features(
    *,
    assignments: Sequence[Sequence[str]],
    mixer_names: list[str],
    feature_config: FeatureExpansionConfig,
) -> np.ndarray:
    """Encode placements into dense feature vectors."""
    type_to_int, _ = build_type_mapping(mixer_names)
    return encode_batch(assignments, type_to_int, feature_config).astype(
        np.float32,
        copy=False,
    )


def build_checkpoint_payload(
    *,
    model: nn.Module,
    reward_model_config: RewardModelConfig,
    optimizer_config: RewardOptimizerConfig,
    mixer_names: list[str],
    feature_config: FeatureExpansionConfig,
    embedding_dim: int,
    placement_feature_dim: int,
    target_mean: float,
    target_std: float,
) -> dict[str, object]:
    """Build a serializable checkpoint payload."""
    return {
        "model_state_dict": model.state_dict(),
        "reward_model_config": reward_model_config.model_dump(),
        "optimizer_config": optimizer_config.model_dump(),
        "mixer_names": mixer_names,
        "feature_config": feature_config.model_dump(),
        "embedding_dim": embedding_dim,
        "placement_feature_dim": placement_feature_dim,
        "target_mean": target_mean,
        "target_std": target_std,
    }


def _activation_module(name: Literal["relu", "gelu", "tanh"]) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    return nn.Tanh()


def _build_mlp(
    *,
    input_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
    activation: Literal["relu", "gelu", "tanh"],
) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(_activation_module(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    return nn.Sequential(*layers), prev_dim


class BaseRewardModel(nn.Module, ABC):
    """Base module for reward-model architectures.

    The prompt encoder may be nonlinear, but the final score is always
    linear in placement features:

        score(prompt, placement) = bias(prompt) + coeffs(prompt)^T phi(placement)
    """

    def __init__(
        self,
        *,
        embedding_dim: int,
        placement_feature_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        activation: Literal["relu", "gelu", "tanh"],
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.placement_feature_dim = placement_feature_dim
        self.prompt_tower, prompt_feature_dim = _build_mlp(
            input_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
        )
        self.coefficient_head = nn.Linear(
            prompt_feature_dim,
            placement_feature_dim,
        )
        self.bias_head = nn.Linear(prompt_feature_dim, 1)

    @abstractmethod
    def pool_prompt(
        self,
        prompt_embeddings: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool token embeddings into one prompt representation."""

    def predict_linear_terms(
        self,
        *,
        prompt_embeddings: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return prompt-conditioned linear coefficients and bias."""
        pooled = self.pool_prompt(prompt_embeddings, prompt_attention_mask)
        prompt_features = self.prompt_tower(pooled)
        coefficients = self.coefficient_head(prompt_features)
        bias = self.bias_head(prompt_features).squeeze(-1)
        return coefficients, bias

    def forward(
        self,
        *,
        prompt_embeddings: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        placement_features: torch.Tensor,
    ) -> torch.Tensor:
        coefficients, bias = self.predict_linear_terms(
            prompt_embeddings=prompt_embeddings,
            prompt_attention_mask=prompt_attention_mask,
        )
        return torch.sum(coefficients * placement_features, dim=-1) + bias


class MeanPoolingRewardModel(BaseRewardModel):
    """Reward model using simple masked mean pooling."""

    def pool_prompt(
        self,
        prompt_embeddings: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = prompt_attention_mask.unsqueeze(-1).to(prompt_embeddings.dtype)
        summed = (prompt_embeddings * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom


class AttentionPoolingRewardModel(BaseRewardModel):
    """Reward model using learned attention pooling."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        placement_feature_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        activation: Literal["relu", "gelu", "tanh"],
        attention_hidden_dim: int,
    ) -> None:
        super().__init__(
            embedding_dim=embedding_dim,
            placement_feature_dim=placement_feature_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
        )
        self.token_attention = nn.Sequential(
            nn.Linear(embedding_dim, attention_hidden_dim),
            _activation_module(activation),
            nn.Linear(attention_hidden_dim, 1),
        )

    def pool_prompt(
        self,
        prompt_embeddings: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = self.token_attention(prompt_embeddings).squeeze(-1)
        scores = scores.masked_fill(~prompt_attention_mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        weights = weights.masked_fill(~prompt_attention_mask, 0.0)
        return torch.sum(prompt_embeddings * weights.unsqueeze(-1), dim=1)


def placement_feature_dim(
    *,
    assignments: Sequence[Sequence[str]],
    mixer_names: Sequence[str],
    feature_config: FeatureExpansionConfig,
) -> int:
    """Compute reward-model placement feature dimension."""
    num_layers = len(assignments[0]) if assignments else 0
    num_types = len(build_type_mapping(list(mixer_names))[0])
    return compute_feature_dim(num_layers, num_types, feature_config)
