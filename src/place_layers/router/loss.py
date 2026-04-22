"""Reward-model fitting objectives."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, Protocol

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from place_layers.router.metrics import _precision_at_k, masked_mean_prompt_rank_correlations

logger = logging.getLogger(__name__)


class RewardModelFactory(Protocol):
    """Config object capable of constructing a reward model."""

    def create_model(
        self,
        *,
        embedding_dim: int,
        placement_feature_dim: int,
    ) -> nn.Module: ...


class RewardOptimizerFactory(Protocol):
    """Config object capable of constructing an optimizer."""

    def create_optimizer(
        self,
        parameters: Sequence[torch.nn.Parameter],
    ) -> torch.optim.Optimizer: ...


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
    train_precision_history: list[float] | None = None


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
        reward_model_config: RewardModelFactory,
        optimizer_config: RewardOptimizerFactory,
        num_epochs: int,
        batch_size: int,
        log_every_n_steps: int,
        precision_k: int,
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
        reward_model_config: RewardModelFactory,
        optimizer_config: RewardOptimizerFactory,
        num_epochs: int,
        batch_size: int,
        log_every_n_steps: int,
        precision_k: int,
        seed: int,
        device: torch.device,
    ) -> RewardModelTrainingArtifacts:
        del precision_k
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be positive")

        prompt_idx_np, placement_idx_np = np.nonzero(np.asarray(observed_mask, dtype=bool))
        if prompt_idx_np.size < 2:
            raise ValueError("Need at least two observed prompt-placement examples to fit reward model")

        targets_np = np.asarray(score_matrix, dtype=np.float32)[prompt_idx_np, placement_idx_np]
        target_mean = float(targets_np.mean())
        target_std = float(targets_np.std())
        target_scale = target_std if target_std >= 1e-6 else 1.0
        standardized_targets = ((targets_np - target_mean) / target_scale).astype(np.float32)

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
                    global_step % log_every_n_steps == 0 or global_step + 1 == total_steps
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
                global_step += 1

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
        reward_model_config: RewardModelFactory,
        optimizer_config: RewardOptimizerFactory,
        num_epochs: int,
        batch_size: int,
        log_every_n_steps: int,
        precision_k: int,
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
        placement_idx_matrix = np.zeros((train_prompt_idx_np.size, max_list_size), dtype=np.int64)
        teacher_score_matrix = np.zeros((train_prompt_idx_np.size, max_list_size), dtype=np.float32)
        list_mask_matrix = np.zeros((train_prompt_idx_np.size, max_list_size), dtype=bool)
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
        precision_history: list[float] = []
        global_step = 0
        num_batches_per_epoch = len(loader)
        total_steps = num_batches_per_epoch * num_epochs
        interval_loss = 0.0
        interval_prompts = 0

        for epoch_idx in range(num_epochs):
            model.train()
            for (batch_row_idx,) in loader:
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
                    global_step % log_every_n_steps == 0 or global_step + 1 == total_steps
                )
                if should_log:
                    y_true = teacher_scores_batch.detach().cpu().numpy().astype(np.float32)
                    y_pred = student_scores.detach().cpu().numpy().astype(np.float32)
                    mask = list_mask_batch.detach().cpu().numpy().astype(bool)
                    mean_spearman, mean_kendall_tau = masked_mean_prompt_rank_correlations(
                        y_true=y_true,
                        y_pred=y_pred,
                        mask=mask,
                    )
                    precision = _precision_at_k(y_true, y_pred, mask, precision_k=precision_k)
                    mean_loss = interval_loss / max(1, interval_prompts)
                    loss_history.append(mean_loss)
                    spearman_history.append(mean_spearman)
                    kendall_tau_history.append(mean_kendall_tau)
                    precision_history.append(precision)
                    message = (
                        "Reward model step "
                        f"{global_step}/{total_steps} "
                        f"(epoch {epoch_idx + 1}/{num_epochs}): "
                        f"train_list_ce={mean_loss:.6f} "
                        f"mean_prompt_spearman={mean_spearman:.4f} "
                        f"mean_prompt_kendall_tau={mean_kendall_tau:.4f} "
                        f"precision@{precision_k}={precision:.4f}"
                    )
                    logger.info(message)
                    print(message, flush=True)
                    interval_loss = 0.0
                    interval_prompts = 0
                global_step += 1

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
            train_precision_history=precision_history,
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
