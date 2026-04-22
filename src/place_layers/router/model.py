"""Torch reward-model utilities for prompt-conditioned placement scoring."""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
import torch
from pydantic import BaseModel, Field
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from place_layers.surrogate.encoding import (
    FeatureExpansionConfig,
    build_type_mapping,
    compute_feature_dim,
    encode_batch,
)

logger = logging.getLogger(__name__)


class RewardModelMetrics(BaseModel):
    """Evaluation metrics for reward-model predictions."""

    rmse: float | None
    r2: float | None
    spearman: float | None
    mean_prompt_spearman: float
    mean_prompt_kendall_tau: float


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


@dataclass(frozen=True)
class RewardModelTrainingArtifacts:
    """Outputs from reward-model training."""

    model: nn.Module
    target_mean: float | None
    target_std: float | None
    train_loss_history: list[float]
    training_objective: str
    teacher_temperature: float | None = None
    student_temperature: float | None = None
    train_mean_prompt_spearman_history: list[float] | None = None
    train_mean_prompt_kendall_tau_history: list[float] | None = None


class BaseRewardLoss(ABC):
    """Runtime reward-loss implementation."""

    def __init__(self, config: "BaseRewardLossConfig") -> None:
        self.config = config

    @abstractmethod
    def fit(
        self,
        *,
        prompt_token_embeddings: np.ndarray,
        prompt_attention_mask: np.ndarray,
        placement_features: np.ndarray,
        score_matrix: np.ndarray,
        observed_mask: np.ndarray,
        reward_model_config: RewardModelConfig,
        optimizer_config: RewardOptimizerConfig,
        num_epochs: int,
        batch_size: int,
        log_every_n_steps: int,
        seed: int,
        device: torch.device,
    ) -> RewardModelTrainingArtifacts:
        """Train a reward model under this loss."""


class MeanSquaredErrorRewardLoss(BaseRewardLoss):
    """Pointwise mean-squared error over observed prompt-placement pairs."""

    config: "MeanSquaredErrorRewardLossConfig"

    def fit(
        self,
        *,
        prompt_token_embeddings: np.ndarray,
        prompt_attention_mask: np.ndarray,
        placement_features: np.ndarray,
        score_matrix: np.ndarray,
        observed_mask: np.ndarray,
        reward_model_config: RewardModelConfig,
        optimizer_config: RewardOptimizerConfig,
        num_epochs: int,
        batch_size: int,
        log_every_n_steps: int,
        seed: int,
        device: torch.device,
    ) -> RewardModelTrainingArtifacts:
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be positive")

        prompt_idx_np, placement_idx_np = np.nonzero(np.asarray(observed_mask, dtype=bool))
        if prompt_idx_np.size < 2:
            raise ValueError("Need at least two observed prompt-placement examples to fit reward model")

        targets_np = np.asarray(score_matrix, dtype=np.float32)[prompt_idx_np, placement_idx_np]
        target_mean = float(targets_np.mean())
        target_std = float(targets_np.std())
        target_scale = target_std if target_std >= 1e-6 else 1.0
        standardized_targets = (
            (targets_np - target_mean) / target_scale
        ).astype(np.float32)

        token_embeddings_tensor = torch.from_numpy(
            np.asarray(prompt_token_embeddings, dtype=np.float32)
        )
        attention_mask_tensor = torch.from_numpy(np.asarray(prompt_attention_mask, dtype=bool))
        placement_features_tensor = torch.from_numpy(
            np.asarray(placement_features, dtype=np.float32)
        )
        prompt_idx_tensor = torch.from_numpy(prompt_idx_np.astype(np.int64))
        placement_idx_tensor = torch.from_numpy(placement_idx_np.astype(np.int64))
        targets_tensor = torch.from_numpy(standardized_targets)

        dataset = TensorDataset(prompt_idx_tensor, placement_idx_tensor, targets_tensor)
        generator = torch.Generator()
        generator.manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        embedding_dim = int(token_embeddings_tensor.shape[-1])
        placement_feature_dim = int(placement_features_tensor.shape[-1])
        model = reward_model_config.create_model(
            embedding_dim=embedding_dim,
            placement_feature_dim=placement_feature_dim,
        ).to(device)
        optimizer = optimizer_config.create_optimizer(list(model.parameters()))
        loss_fn = nn.MSELoss()

        loss_history: list[float] = []
        global_step = 0
        num_batches_per_epoch = len(loader)
        total_steps = num_batches_per_epoch * num_epochs
        interval_loss = 0.0
        interval_examples = 0

        for epoch_idx in range(num_epochs):
            model.train()
            for prompt_idx_batch, placement_idx_batch, target_batch in loader:
                global_step += 1
                target_batch = target_batch.to(device)
                prompt_embeddings_batch = token_embeddings_tensor[prompt_idx_batch].to(device)
                prompt_attention_mask_batch = attention_mask_tensor[prompt_idx_batch].to(device)
                placement_features_batch = placement_features_tensor[placement_idx_batch].to(device)

                optimizer.zero_grad(set_to_none=True)
                preds = model(
                    prompt_embeddings=prompt_embeddings_batch,
                    prompt_attention_mask=prompt_attention_mask_batch,
                    placement_features=placement_features_batch,
                )
                loss = loss_fn(preds, target_batch)
                loss.backward()
                optimizer.step()

                batch_size_actual = int(target_batch.shape[0])
                interval_loss += float(loss.item()) * batch_size_actual
                interval_examples += batch_size_actual

                should_log = (
                    global_step % log_every_n_steps == 0 or global_step == total_steps
                )
                if should_log:
                    mean_loss = interval_loss / max(1, interval_examples)
                    loss_history.append(mean_loss)
                    message = (
                        "Reward model step "
                        f"{global_step}/{total_steps} "
                        f"(epoch {epoch_idx + 1}/{num_epochs}): "
                        f"train_mse={mean_loss:.6f}"
                    )
                    logger.info(message)
                    print(message, flush=True)
                    interval_loss = 0.0
                    interval_examples = 0

        return RewardModelTrainingArtifacts(
            model=model,
            target_mean=target_mean,
            target_std=target_std,
            train_loss_history=loss_history,
            training_objective=self.config.loss_type,
        )


class ListCrossEntropyRewardLoss(BaseRewardLoss):
    """Listwise cross-entropy over observed placements for each prompt."""

    config: "ListCrossEntropyRewardLossConfig"

    def fit(
        self,
        *,
        prompt_token_embeddings: np.ndarray,
        prompt_attention_mask: np.ndarray,
        placement_features: np.ndarray,
        score_matrix: np.ndarray,
        observed_mask: np.ndarray,
        reward_model_config: RewardModelConfig,
        optimizer_config: RewardOptimizerConfig,
        num_epochs: int,
        batch_size: int,
        log_every_n_steps: int,
        seed: int,
        device: torch.device,
    ) -> RewardModelTrainingArtifacts:
        teacher_temperature = self.config.teacher_temperature
        student_temperature = self.config.student_temperature
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be positive")
        if teacher_temperature <= 0.0:
            raise ValueError("teacher_temperature must be positive")
        if student_temperature <= 0.0:
            raise ValueError("student_temperature must be positive")

        observed_mask_np = np.asarray(observed_mask, dtype=bool)
        observed_scores = np.asarray(score_matrix, dtype=np.float32)[observed_mask_np]
        if observed_scores.size == 0:
            raise ValueError("Need at least one observed prompt-placement example to fit reward model")
        target_mean = float(observed_scores.mean())
        target_std = float(observed_scores.std())
        observed_counts = observed_mask_np.sum(axis=1)
        train_prompt_idx_np = np.flatnonzero(observed_counts >= 2)
        if train_prompt_idx_np.size == 0:
            raise ValueError(
                "Need at least one prompt with at least two observed placements to fit "
                "the listwise reward model"
            )

        max_list_size = int(observed_counts[train_prompt_idx_np].max())
        placement_idx_matrix = np.zeros(
            (train_prompt_idx_np.size, max_list_size),
            dtype=np.int64,
        )
        teacher_score_matrix = np.zeros(
            (train_prompt_idx_np.size, max_list_size),
            dtype=np.float32,
        )
        list_mask_matrix = np.zeros(
            (train_prompt_idx_np.size, max_list_size),
            dtype=bool,
        )
        score_matrix_np = np.asarray(score_matrix, dtype=np.float32)
        for row_idx, prompt_idx in enumerate(train_prompt_idx_np):
            placement_idx = np.flatnonzero(observed_mask_np[prompt_idx])
            list_size = int(placement_idx.size)
            placement_idx_matrix[row_idx, :list_size] = placement_idx
            teacher_score_matrix[row_idx, :list_size] = score_matrix_np[prompt_idx, placement_idx]
            list_mask_matrix[row_idx, :list_size] = True

        token_embeddings_tensor = torch.from_numpy(
            np.asarray(prompt_token_embeddings, dtype=np.float32)
        )
        attention_mask_tensor = torch.from_numpy(np.asarray(prompt_attention_mask, dtype=bool))
        placement_features_tensor = torch.from_numpy(
            np.asarray(placement_features, dtype=np.float32)
        )
        train_prompt_idx_tensor = torch.from_numpy(train_prompt_idx_np.astype(np.int64))
        placement_idx_tensor = torch.from_numpy(placement_idx_matrix)
        teacher_scores_tensor = torch.from_numpy(teacher_score_matrix)
        list_mask_tensor = torch.from_numpy(list_mask_matrix)

        dataset = TensorDataset(torch.arange(train_prompt_idx_np.size, dtype=torch.int64))
        generator = torch.Generator()
        generator.manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        embedding_dim = int(token_embeddings_tensor.shape[-1])
        placement_feature_dim = int(placement_features_tensor.shape[-1])
        model = reward_model_config.create_model(
            embedding_dim=embedding_dim,
            placement_feature_dim=placement_feature_dim,
        ).to(device)
        optimizer = optimizer_config.create_optimizer(list(model.parameters()))

        loss_history: list[float] = []
        spearman_history: list[float] = []
        kendall_tau_history: list[float] = []
        global_step = 0
        num_batches_per_epoch = len(loader)
        total_steps = num_batches_per_epoch * num_epochs
        interval_loss = 0.0
        interval_prompts = 0

        for epoch_idx in range(num_epochs):
            model.train()
            for (batch_row_idx,) in loader:
                global_step += 1
                prompt_idx_batch = train_prompt_idx_tensor[batch_row_idx]
                prompt_embeddings_batch = token_embeddings_tensor[prompt_idx_batch].to(device)
                prompt_attention_mask_batch = attention_mask_tensor[prompt_idx_batch].to(device)
                placement_idx_batch = placement_idx_tensor[batch_row_idx]
                teacher_scores_batch = teacher_scores_tensor[batch_row_idx].to(device)
                list_mask_batch = list_mask_tensor[batch_row_idx].to(device)

                optimizer.zero_grad(set_to_none=True)
                coefficients, bias = model.predict_linear_terms(
                    prompt_embeddings=prompt_embeddings_batch,
                    prompt_attention_mask=prompt_attention_mask_batch,
                )
                placement_features_batch = placement_features_tensor[placement_idx_batch].to(device)
                student_scores = torch.einsum(
                    "bld,bd->bl",
                    placement_features_batch,
                    coefficients,
                ) + bias.unsqueeze(-1)
                student_logits = (student_scores / student_temperature).masked_fill(
                    ~list_mask_batch,
                    float("-inf"),
                )
                teacher_logits = (teacher_scores_batch / teacher_temperature).masked_fill(
                    ~list_mask_batch,
                    float("-inf"),
                )
                teacher_probs = F.softmax(teacher_logits, dim=-1)
                log_student_probs = F.log_softmax(student_logits, dim=-1)
                teacher_probs = teacher_probs.masked_fill(~list_mask_batch, 0.0)
                log_student_probs = log_student_probs.masked_fill(~list_mask_batch, 0.0)
                loss = -(teacher_probs * log_student_probs).sum(dim=-1).mean()
                loss.backward()
                optimizer.step()

                batch_prompt_count = int(prompt_idx_batch.shape[0])
                interval_loss += float(loss.item()) * batch_prompt_count
                interval_prompts += batch_prompt_count

                should_log = (
                    global_step % log_every_n_steps == 0 or global_step == total_steps
                )
                if should_log:
                    mean_spearman, mean_kendall_tau = _masked_mean_prompt_rank_correlations(
                        y_true=teacher_scores_batch.detach().cpu().numpy().astype(np.float32),
                        y_pred=student_scores.detach().cpu().numpy().astype(np.float32),
                        mask=list_mask_batch.detach().cpu().numpy().astype(bool),
                    )
                    mean_loss = interval_loss / max(1, interval_prompts)
                    loss_history.append(mean_loss)
                    spearman_history.append(mean_spearman)
                    kendall_tau_history.append(mean_kendall_tau)
                    message = (
                        "Reward model step "
                        f"{global_step}/{total_steps} "
                        f"(epoch {epoch_idx + 1}/{num_epochs}): "
                        f"train_list_ce={mean_loss:.6f} "
                        f"mean_prompt_spearman={mean_spearman:.4f} "
                        f"mean_prompt_kendall_tau={mean_kendall_tau:.4f}"
                    )
                    logger.info(message)
                    print(message, flush=True)
                    interval_loss = 0.0
                    interval_prompts = 0

        return RewardModelTrainingArtifacts(
            model=model,
            target_mean=target_mean,
            target_std=target_std,
            train_loss_history=loss_history,
            training_objective=self.config.loss_type,
            teacher_temperature=self.config.teacher_temperature,
            student_temperature=self.config.student_temperature,
            train_mean_prompt_spearman_history=spearman_history,
            train_mean_prompt_kendall_tau_history=kendall_tau_history,
        )


class BaseRewardLossConfig(BaseModel):
    """Base config for reward-model training losses."""

    loss_type: str = Field(description="Loss discriminator for reward-model fitting.")
    _impl_class: ClassVar[type[BaseRewardLoss]]

    def create_loss(self) -> BaseRewardLoss:
        return self._impl_class(config=self)


class MeanSquaredErrorRewardLossConfig(BaseRewardLossConfig):
    """Pointwise MSE reward-model fitting."""

    _impl_class: ClassVar[type[BaseRewardLoss]] = MeanSquaredErrorRewardLoss

    loss_type: Literal["mse_loss"] = "mse_loss"


class ListCrossEntropyRewardLossConfig(BaseRewardLossConfig):
    """Listwise cross-entropy reward-model fitting."""

    _impl_class: ClassVar[type[BaseRewardLoss]] = ListCrossEntropyRewardLoss

    loss_type: Literal["list_cross_entropy_loss"] = "list_cross_entropy_loss"
    teacher_temperature: float = Field(
        default=1.0,
        gt=0.0,
        description="Temperature used for the teacher placement distribution.",
    )
    student_temperature: float = Field(
        default=1.0,
        gt=0.0,
        description="Temperature used for the student placement distribution.",
    )


RewardLossConfig = Annotated[
    MeanSquaredErrorRewardLossConfig | ListCrossEntropyRewardLossConfig,
    Field(discriminator="loss_type"),
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


def evaluate_reward_model(
    model: nn.Module,
    *,
    prompt_token_embeddings: np.ndarray,
    prompt_attention_mask: np.ndarray,
    placement_features: np.ndarray,
    score_matrix: np.ndarray,
    observed_mask: np.ndarray,
    batch_size: int,
    device: torch.device,
    target_mean: float,
    target_std: float,
    report_pointwise_metrics: bool = True,
) -> tuple[RewardModelMetrics, np.ndarray]:
    """Evaluate a reward model on observed prompt-placement pairs."""
    prompt_idx_np, placement_idx_np = np.nonzero(np.asarray(observed_mask, dtype=bool))
    preds_matrix = np.zeros_like(score_matrix, dtype=np.float32)
    if prompt_idx_np.size == 0:
        empty_metrics = RewardModelMetrics(
            rmse=0.0 if report_pointwise_metrics else None,
            r2=0.0 if report_pointwise_metrics else None,
            spearman=0.0 if report_pointwise_metrics else None,
            mean_prompt_spearman=0.0,
            mean_prompt_kendall_tau=0.0,
        )
        return empty_metrics, preds_matrix

    raw_preds_matrix = _predict_reward_model_matrix(
        model,
        prompt_token_embeddings=prompt_token_embeddings,
        prompt_attention_mask=prompt_attention_mask,
        placement_features=placement_features,
        observed_mask=observed_mask,
        batch_size=batch_size,
        device=device,
    )
    if not report_pointwise_metrics:
        metrics = RewardModelMetrics(
            rmse=None,
            r2=None,
            spearman=None,
            mean_prompt_spearman=_mean_prompt_spearman(
                np.asarray(score_matrix, dtype=np.float32),
                raw_preds_matrix,
                np.asarray(observed_mask, dtype=bool),
            ),
            mean_prompt_kendall_tau=_mean_prompt_kendall_tau(
                np.asarray(score_matrix, dtype=np.float32),
                raw_preds_matrix,
                np.asarray(observed_mask, dtype=bool),
            ),
        )
        return metrics, raw_preds_matrix

    target_scale = target_std if target_std >= 1e-6 else 1.0
    preds_matrix = raw_preds_matrix * target_scale + target_mean
    pred_values = preds_matrix[prompt_idx_np, placement_idx_np]
    y_true = np.asarray(score_matrix, dtype=np.float32)[prompt_idx_np, placement_idx_np]
    mse = float(mean_squared_error(y_true, pred_values))
    metrics = RewardModelMetrics(
        rmse=float(math.sqrt(mse)),
        r2=float(r2_score(y_true, pred_values)) if y_true.size >= 2 else 0.0,
        spearman=_safe_spearman(y_true, pred_values),
        mean_prompt_spearman=_mean_prompt_spearman(
            np.asarray(score_matrix, dtype=np.float32),
            preds_matrix,
            np.asarray(observed_mask, dtype=bool),
        ),
        mean_prompt_kendall_tau=_mean_prompt_kendall_tau(
            np.asarray(score_matrix, dtype=np.float32),
            preds_matrix,
            np.asarray(observed_mask, dtype=bool),
        ),
    )
    return metrics, preds_matrix


def _predict_reward_model_matrix(
    model: nn.Module,
    *,
    prompt_token_embeddings: np.ndarray,
    prompt_attention_mask: np.ndarray,
    placement_features: np.ndarray,
    observed_mask: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Predict raw reward-model outputs on all observed prompt-placement pairs."""
    prompt_idx_np, placement_idx_np = np.nonzero(np.asarray(observed_mask, dtype=bool))
    preds_matrix = np.zeros(np.asarray(observed_mask).shape, dtype=np.float32)
    if prompt_idx_np.size == 0:
        return preds_matrix

    token_embeddings_tensor = torch.from_numpy(
        np.asarray(prompt_token_embeddings, dtype=np.float32)
    )
    attention_mask_tensor = torch.from_numpy(np.asarray(prompt_attention_mask, dtype=bool))
    placement_features_tensor = torch.from_numpy(
        np.asarray(placement_features, dtype=np.float32)
    )

    model.eval()
    pred_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, prompt_idx_np.size, batch_size):
            end = min(start + batch_size, prompt_idx_np.size)
            prompt_idx_batch = torch.from_numpy(prompt_idx_np[start:end].astype(np.int64))
            placement_idx_batch = torch.from_numpy(
                placement_idx_np[start:end].astype(np.int64)
            )
            batch_preds = model(
                prompt_embeddings=token_embeddings_tensor[prompt_idx_batch].to(device),
                prompt_attention_mask=attention_mask_tensor[prompt_idx_batch].to(device),
                placement_features=placement_features_tensor[placement_idx_batch].to(device),
            )
            pred_chunks.append(batch_preds.detach().cpu().numpy().astype(np.float32))

    preds_matrix[prompt_idx_np, placement_idx_np] = np.concatenate(pred_chunks, axis=0)
    return preds_matrix


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


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return 0.0
    rho, _ = spearmanr(y_true, y_pred)
    return 0.0 if np.isnan(rho) else float(rho)


def _safe_kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return 0.0
    tau, _ = kendalltau(y_true, y_pred)
    return 0.0 if np.isnan(tau) else float(tau)


def _mean_prompt_spearman(
    y_true_matrix: np.ndarray,
    y_pred_matrix: np.ndarray,
    observed_mask: np.ndarray,
) -> float:
    if observed_mask.shape != y_true_matrix.shape:
        raise ValueError("observed_mask must have the same shape as score_matrix")

    rhos: list[float] = []
    for y_true_row, y_pred_row, row_mask in zip(
        y_true_matrix,
        y_pred_matrix,
        observed_mask,
        strict=True,
    ):
        rhos.append(_safe_spearman(y_true_row[row_mask], y_pred_row[row_mask]))
    return float(np.mean(rhos)) if rhos else 0.0


def _mean_prompt_kendall_tau(
    y_true_matrix: np.ndarray,
    y_pred_matrix: np.ndarray,
    observed_mask: np.ndarray,
) -> float:
    if observed_mask.shape != y_true_matrix.shape:
        raise ValueError("observed_mask must have the same shape as score_matrix")

    taus: list[float] = []
    for y_true_row, y_pred_row, row_mask in zip(
        y_true_matrix,
        y_pred_matrix,
        observed_mask,
        strict=True,
    ):
        taus.append(_safe_kendall_tau(y_true_row[row_mask], y_pred_row[row_mask]))
    return float(np.mean(taus)) if taus else 0.0


def _masked_mean_prompt_rank_correlations(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    if y_true.shape != y_pred.shape or y_true.shape != mask.shape:
        raise ValueError("y_true, y_pred, and mask must have the same shape")

    spearmans: list[float] = []
    kendall_taus: list[float] = []
    for y_true_row, y_pred_row, row_mask in zip(y_true, y_pred, mask, strict=True):
        spearmans.append(_safe_spearman(y_true_row[row_mask], y_pred_row[row_mask]))
        kendall_taus.append(_safe_kendall_tau(y_true_row[row_mask], y_pred_row[row_mask]))
    if not spearmans:
        return 0.0, 0.0
    return float(np.mean(spearmans)), float(np.mean(kendall_taus))


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
