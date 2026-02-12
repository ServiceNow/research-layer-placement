"""Surrogate model selection for placement optimization.

Provides automatic expansion level selection via RidgeCV to determine
the optimal cluster expansion depth (unary, pairwise range, interaction
order) using a two-phase search.
"""

from __future__ import annotations

import logging

import numpy as np
from pydantic import BaseModel
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from place_layers.surrogate.encoding import (
    FeatureExpansionConfig,
    build_type_mapping,
    compute_feature_dim,
    encode_batch,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class StepResult(BaseModel):
    """One step in the range or order search."""

    model_config = {"frozen": True}

    config: FeatureExpansionConfig
    num_features: int
    rho: float


class ExpansionSelectionResult(BaseModel):
    """Result of two-phase expansion selection."""

    model_config = {"frozen": True}

    selected_config: FeatureExpansionConfig
    selected_range: int
    selected_order: int
    range_results: list[StepResult]
    order_results: list[StepResult]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ridge_cv_spearman(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    seed: int,
) -> float:
    """Compute mean Spearman rho via k-fold CV using RidgeCV.

    RidgeCV is used for expansion level selection because it's fast
    and appropriate for comparing linear-in-indicators models.
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    rhos: list[float] = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        ridge = RidgeCV(alphas=np.logspace(-3, 6, 10), gcv_mode="svd")
        ridge.fit(X_train, y_train)
        preds = ridge.predict(X_val)

        if len(y_val) >= 2:
            rho, _ = spearmanr(y_val, preds)
            if np.isnan(rho):
                rho = 0.0
            rhos.append(rho)

    return float(np.mean(rhos)) if rhos else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ridge_cv_rho(
    assignments: list[tuple[str, ...]],
    scores: np.ndarray,
    mixer_names: list[str],
    feature_config: FeatureExpansionConfig | None = None,
    cv_folds: int = 5,
    seed: int = 42,
) -> float:
    """Compute Ridge CV Spearman rho for the adaptive exploration loop.

    Thin wrapper around :func:`_ridge_cv_spearman` that handles encoding.

    Args:
        assignments: Evaluated placements.
        scores: Corresponding scores, shape ``(n,)``.
        mixer_names: Mixer type names for encoding.
        feature_config: Optional feature expansion config.
        cv_folds: Number of CV folds.
        seed: Random seed.

    Returns:
        Mean Spearman rho across folds.
    """
    type_to_int, _ = build_type_mapping(mixer_names)
    X = encode_batch(assignments, type_to_int, feature_config)
    return _ridge_cv_spearman(X, scores, cv_folds, seed)


def select_expansion_level(
    assignments: list[tuple[str, ...]],
    scores: np.ndarray,
    mixer_names: list[str],
    cv_folds: int = 5,
    target_rho: float = 0.95,
    min_improvement: float = 0.01,
    max_pairwise_range: int = 10,
    max_order: int = 3,
    seed: int = 42,
    min_rho_warn: float = 0.3,
) -> ExpansionSelectionResult:
    """Automatic expansion selection using a two-phase search.

    Phase 1 (range selection): At order 2, sweep pairwise range from 1 to
    *max_pairwise_range* to find the best range. A unary baseline (order 1)
    is always computed first.

    Phase 2 (order selection): At the selected range, sweep interaction
    order from 3 to *max_order*.

    Each phase stops early when *target_rho* is reached, marginal
    improvement drops below *min_improvement*, or the feature-to-sample
    ratio exceeds 2:1.

    Args:
        assignments: Evaluated placements.
        scores: Corresponding scores, shape ``(n,)``.
        mixer_names: Mixer type names for encoding.
        cv_folds: Number of CV folds.
        target_rho: Target Spearman rho for early termination.
        min_improvement: Minimum rho improvement to keep climbing.
        max_pairwise_range: Ceiling for pairwise skip distance.
        max_order: Maximum n-body order to test.
        seed: Random seed.
        min_rho_warn: Warn if final rho is below this threshold.

    Returns:
        :class:`ExpansionSelectionResult` with selected config and
        per-phase diagnostics.
    """
    type_to_int, _ = build_type_mapping(mixer_names)
    num_layers = len(assignments[0]) if assignments else 0
    num_types = len(type_to_int)
    n_samples = len(assignments)

    # --- Unary baseline ---
    unary_config = FeatureExpansionConfig(max_order=1)
    unary_features = compute_feature_dim(num_layers, num_types, unary_config)
    X_unary = encode_batch(assignments, type_to_int, unary_config)
    unary_rho = _ridge_cv_spearman(X_unary, scores, cv_folds, seed)

    range_results: list[StepResult] = [
        StepResult(config=unary_config, num_features=unary_features, rho=unary_rho),
    ]

    best_range = 0
    best_order = 1
    prev_rho = unary_rho

    # If max_order < 2, return early with unary only
    if max_order < 2 or unary_rho >= target_rho:
        selected_config = unary_config
        if unary_rho < min_rho_warn:
            logger.warning(
                "Low surrogate quality: rho=%.4g (threshold=%.4g). "
                "Consider collecting more samples.",
                unary_rho,
                min_rho_warn,
            )
        return ExpansionSelectionResult(
            selected_config=selected_config,
            selected_range=0,
            selected_order=1,
            range_results=range_results,
            order_results=[],
        )

    # --- Phase 1: Range selection at order 2 ---
    for r in range(1, max_pairwise_range + 1):
        config = FeatureExpansionConfig(max_order=2, pairwise_max_range=r)
        num_features = compute_feature_dim(num_layers, num_types, config)

        # Feature-to-sample guard
        if num_features > 2 * n_samples:
            break

        X = encode_batch(assignments, type_to_int, config)
        rho = _ridge_cv_spearman(X, scores, cv_folds, seed)

        range_results.append(
            StepResult(config=config, num_features=num_features, rho=rho),
        )

        # Target reached — select this range
        if rho >= target_rho:
            best_range = r
            prev_rho = rho
            break

        # Marginal improvement too small — keep previous
        if (rho - prev_rho) < min_improvement:
            break

        best_range = r
        prev_rho = rho

    # Determine selected config after phase 1
    best_order = 2 if best_range >= 1 else 1
    if best_order == 1:
        selected_config = unary_config
    else:
        selected_config = FeatureExpansionConfig(
            max_order=2, pairwise_max_range=best_range,
        )

    # --- Phase 2: Order selection at selected range ---
    order_results: list[StepResult] = []

    if best_range >= 1 and max_order >= 3 and prev_rho < target_rho:
        for n in range(3, max_order + 1):
            config = FeatureExpansionConfig(
                max_order=n, pairwise_max_range=best_range,
            )
            num_features = compute_feature_dim(num_layers, num_types, config)

            # Feature-to-sample guard
            if num_features > 2 * n_samples:
                break

            X = encode_batch(assignments, type_to_int, config)
            rho = _ridge_cv_spearman(X, scores, cv_folds, seed)

            order_results.append(
                StepResult(config=config, num_features=num_features, rho=rho),
            )

            # Target reached
            if rho >= target_rho:
                best_order = n
                selected_config = config
                prev_rho = rho
                break

            # Marginal improvement too small — keep previous
            if (rho - prev_rho) < min_improvement:
                break

            best_order = n
            selected_config = config
            prev_rho = rho

    # --- Warning for low rho ---
    if prev_rho < min_rho_warn:
        logger.warning(
            "Low surrogate quality: rho=%.4g (threshold=%.4g). "
            "Consider collecting more samples.",
            prev_rho,
            min_rho_warn,
        )

    return ExpansionSelectionResult(
        selected_config=selected_config,
        selected_range=best_range,
        selected_order=best_order,
        range_results=range_results,
        order_results=order_results,
    )
