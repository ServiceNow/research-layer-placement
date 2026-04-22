"""Reward-model evaluation metrics and helpers."""

from __future__ import annotations

import math

import numpy as np
import torch
from pydantic import BaseModel
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from torch import nn


class RewardModelMetrics(BaseModel):
    """Evaluation metrics for reward-model predictions."""

    rmse: float | None
    r2: float | None
    spearman: float | None
    mean_prompt_spearman: float
    mean_prompt_kendall_tau: float
    precision: float


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
    precision_k: int,
    report_pointwise_metrics: bool = True,
) -> tuple[RewardModelMetrics, np.ndarray]:
    """Evaluate a reward model on observed prompt-placement pairs."""
    if precision_k <= 0:
        raise ValueError("precision_k must be positive")
    prompt_idx_np, placement_idx_np = np.nonzero(np.asarray(observed_mask, dtype=bool))
    preds_matrix = np.zeros_like(score_matrix, dtype=np.float32)
    if prompt_idx_np.size == 0:
        empty_metrics = RewardModelMetrics(
            rmse=0.0 if report_pointwise_metrics else None,
            r2=0.0 if report_pointwise_metrics else None,
            spearman=0.0 if report_pointwise_metrics else None,
            mean_prompt_spearman=0.0,
            mean_prompt_kendall_tau=0.0,
            precision=0.0,
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
    score_matrix_np = np.asarray(score_matrix, dtype=np.float32)
    observed_mask_np = np.asarray(observed_mask, dtype=bool)
    if not report_pointwise_metrics:
        metrics = RewardModelMetrics(
            rmse=None,
            r2=None,
            spearman=None,
            mean_prompt_spearman=_mean_prompt_spearman(
                score_matrix_np,
                raw_preds_matrix,
                observed_mask_np,
            ),
            mean_prompt_kendall_tau=_mean_prompt_kendall_tau(
                score_matrix_np,
                raw_preds_matrix,
                observed_mask_np,
            ),
            precision=_precision_at_k(
                score_matrix_np,
                raw_preds_matrix,
                observed_mask_np,
                precision_k=precision_k,
            ),
        )
        return metrics, raw_preds_matrix

    target_scale = target_std if target_std >= 1e-6 else 1.0
    preds_matrix = raw_preds_matrix * target_scale + target_mean
    pred_values = preds_matrix[prompt_idx_np, placement_idx_np]
    y_true = score_matrix_np[prompt_idx_np, placement_idx_np]
    mse = float(mean_squared_error(y_true, pred_values))
    metrics = RewardModelMetrics(
        rmse=float(math.sqrt(mse)),
        r2=float(r2_score(y_true, pred_values)) if y_true.size >= 2 else 0.0,
        spearman=_safe_spearman(y_true, pred_values),
        mean_prompt_spearman=_mean_prompt_spearman(
            score_matrix_np,
            preds_matrix,
            observed_mask_np,
        ),
        mean_prompt_kendall_tau=_mean_prompt_kendall_tau(
            score_matrix_np,
            preds_matrix,
            observed_mask_np,
        ),
        precision=_precision_at_k(
            score_matrix_np,
            preds_matrix,
            observed_mask_np,
            precision_k=precision_k,
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


def masked_mean_prompt_rank_correlations(
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


def _precision_at_k(
    y_true_matrix: np.ndarray,
    y_pred_matrix: np.ndarray,
    observed_mask: np.ndarray,
    *,
    precision_k: int,
) -> float:
    if observed_mask.shape != y_true_matrix.shape:
        raise ValueError("observed_mask must have the same shape as score_matrix")
    if y_pred_matrix.shape != y_true_matrix.shape:
        raise ValueError("y_pred_matrix must have the same shape as score_matrix")

    hits = 0
    total = 0
    for y_true_row, y_pred_row, row_mask in zip(
        y_true_matrix,
        y_pred_matrix,
        observed_mask,
        strict=True,
    ):
        observed_idx = np.flatnonzero(row_mask)
        if observed_idx.size == 0:
            continue
        total += 1
        predicted_top_idx = observed_idx[np.argmax(y_pred_row[observed_idx])]
        top_k = min(precision_k, int(observed_idx.size))
        top_true_idx = observed_idx[
            np.argpartition(y_true_row[observed_idx], -top_k)[-top_k:]
        ]
        if predicted_top_idx in set(top_true_idx.tolist()):
            hits += 1
    return float(hits / total) if total > 0 else 0.0
