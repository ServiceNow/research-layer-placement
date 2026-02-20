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
    FeatureScaling,
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
    score: float
    rho: float | None = None
    stop_reason: str | None = None  # None = intermediate step; set on terminal step


class ExpansionSelectionResult(BaseModel):
    """Result of two-phase expansion selection."""

    model_config = {"frozen": True}

    selected_config: FeatureExpansionConfig
    selected_range: int
    selected_order: int
    range_results: list[StepResult]
    order_results: list[StepResult]
    phase1_stop_reason: str | None = None  # Why range search stopped
    phase2_stop_reason: str | None = None  # Why order search stopped (None if skipped)


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

        ridge = RidgeCV(alphas=np.logspace(-3, 6, 10))
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


def select_expansion_ridge_cv(
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
        StepResult(config=unary_config, num_features=unary_features, score=unary_rho, rho=unary_rho),
    ]

    best_range = 0
    best_order = 1
    prev_rho = unary_rho

    # If max_order < 2, return early with unary only
    if max_order < 2 or unary_rho >= target_rho:
        early_reason = "target_reached" if unary_rho >= target_rho else None
        if early_reason:
            range_results[-1] = range_results[-1].model_copy(
                update={"stop_reason": early_reason},
            )
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
            phase1_stop_reason=early_reason,
        )

    # --- Phase 1: Range selection at order 2 ---
    phase1_stop_reason: str | None = None
    for r in range(1, max_pairwise_range + 1):
        config = FeatureExpansionConfig(max_order=2, pairwise_max_range=r)
        num_features = compute_feature_dim(num_layers, num_types, config)

        # Feature-to-sample guard
        if num_features > 2 * n_samples:
            phase1_stop_reason = "feature_guard"
            range_results[-1] = range_results[-1].model_copy(
                update={"stop_reason": "feature_guard"},
            )
            break

        X = encode_batch(assignments, type_to_int, config)
        rho = _ridge_cv_spearman(X, scores, cv_folds, seed)

        range_results.append(
            StepResult(config=config, num_features=num_features, score=rho, rho=rho),
        )

        # Target reached — select this range
        if rho >= target_rho:
            phase1_stop_reason = "target_reached"
            range_results[-1] = range_results[-1].model_copy(
                update={"stop_reason": "target_reached"},
            )
            best_range = r
            prev_rho = rho
            break

        # Marginal improvement too small — keep previous
        if (rho - prev_rho) < min_improvement:
            phase1_stop_reason = "below_min_improvement"
            range_results[-1] = range_results[-1].model_copy(
                update={"stop_reason": "below_min_improvement"},
            )
            break

        best_range = r
        prev_rho = rho

    # Loop exhausted without early stop
    if phase1_stop_reason is None and range_results:
        phase1_stop_reason = "max_range"
        range_results[-1] = range_results[-1].model_copy(
            update={"stop_reason": "max_range"},
        )

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
    phase2_stop_reason: str | None = None

    if best_range >= 1 and max_order >= 3 and prev_rho < target_rho:
        for n in range(3, max_order + 1):
            config = FeatureExpansionConfig(
                max_order=n, pairwise_max_range=best_range,
            )
            num_features = compute_feature_dim(num_layers, num_types, config)

            # Feature-to-sample guard
            if num_features > 2 * n_samples:
                phase2_stop_reason = "feature_guard"
                # Mark the last evaluated step (could be from range_results
                # if no order step ran yet, but we only annotate order_results)
                if order_results:
                    order_results[-1] = order_results[-1].model_copy(
                        update={"stop_reason": "feature_guard"},
                    )
                break

            X = encode_batch(assignments, type_to_int, config)
            rho = _ridge_cv_spearman(X, scores, cv_folds, seed)

            order_results.append(
                StepResult(config=config, num_features=num_features, score=rho, rho=rho),
            )

            # Target reached
            if rho >= target_rho:
                phase2_stop_reason = "target_reached"
                order_results[-1] = order_results[-1].model_copy(
                    update={"stop_reason": "target_reached"},
                )
                best_order = n
                selected_config = config
                prev_rho = rho
                break

            # Marginal improvement too small — keep previous
            if (rho - prev_rho) < min_improvement:
                phase2_stop_reason = "below_min_improvement"
                order_results[-1] = order_results[-1].model_copy(
                    update={"stop_reason": "below_min_improvement"},
                )
                break

            best_order = n
            selected_config = config
            prev_rho = rho

        # Loop exhausted without early stop
        if phase2_stop_reason is None and order_results:
            phase2_stop_reason = "max_order"
            order_results[-1] = order_results[-1].model_copy(
                update={"stop_reason": "max_order"},
            )

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
        phase1_stop_reason=phase1_stop_reason,
        phase2_stop_reason=phase2_stop_reason,
    )


# ---------------------------------------------------------------------------
# Bayesian evidence-based expansion selection
# ---------------------------------------------------------------------------


def _bayesian_log_evidence(
    assignments: list[tuple[str, ...]],
    y: np.ndarray,
    mixer_names: list[str],
    feature_config: FeatureExpansionConfig,
    prior_alpha: float,
    prior_a0: float,
    prior_b0: float,
    feature_scaling: FeatureScaling = FeatureScaling.COUNTS_ONLY,
) -> float:
    """Compute log marginal likelihood under a BayesianRidgeSurrogate.

    Deterministic — no CV splits, no RNG in fit.
    """
    from place_layers.surrogate.bayesian_ridge import BayesianRidgeSurrogate

    surrogate = BayesianRidgeSurrogate(
        mixer_names=mixer_names,
        seed=0,  # unused — fit is deterministic
        feature_config=feature_config,
        prior_alpha=prior_alpha,
        prior_a0=prior_a0,
        prior_b0=prior_b0,
        feature_scaling=feature_scaling,
    )
    surrogate.fit(assignments, y)
    return surrogate.log_marginal_likelihood()


def select_expansion_bayes_evidence(
    assignments: list[tuple[str, ...]],
    scores: np.ndarray,
    mixer_names: list[str],
    prior_alpha: float,
    prior_a0: float,
    prior_b0: float,
    feature_scaling: FeatureScaling = FeatureScaling.COUNTS_ONLY,
    max_pairwise_range: int = 10,
    max_order: int = 3,
) -> ExpansionSelectionResult:
    """Expansion selection via Bayesian log marginal likelihood (evidence).

    Enumerates all feasible (order, range) combos with a feature-to-sample
    guard (d > 2*n → skip), computes evidence for each, returns the argmax.

    Fully deterministic — no seed parameter needed.
    """
    type_to_int, _ = build_type_mapping(mixer_names)
    num_layers = len(assignments[0]) if assignments else 0
    num_types = len(type_to_int)
    n_samples = len(assignments)

    all_candidates: list[StepResult] = []

    def _eval(config: FeatureExpansionConfig) -> StepResult | None:
        num_features = compute_feature_dim(num_layers, num_types, config)
        if num_features > 2 * n_samples:
            return None
        evidence = _bayesian_log_evidence(
            assignments, scores, mixer_names, config,
            prior_alpha, prior_a0, prior_b0, feature_scaling,
        )
        return StepResult(
            config=config, num_features=num_features, score=evidence, rho=None,
        )

    # Order 1 (unary)
    unary_config = FeatureExpansionConfig(max_order=1)
    unary_step = _eval(unary_config)
    if unary_step is not None:
        all_candidates.append(unary_step)

    range_results: list[StepResult] = list(all_candidates)
    order_results: list[StepResult] = []

    # Order 2: sweep pairwise range
    for r in range(1, max_pairwise_range + 1):
        config = FeatureExpansionConfig(max_order=2, pairwise_max_range=r)
        step = _eval(config)
        if step is None:
            break
        all_candidates.append(step)
        range_results.append(step)

    # Orders 3..max_order: sweep pairwise range for each
    for n in range(3, max_order + 1):
        for r in range(1, max_pairwise_range + 1):
            config = FeatureExpansionConfig(max_order=n, pairwise_max_range=r)
            step = _eval(config)
            if step is None:
                break
            all_candidates.append(step)
            order_results.append(step)

    # Select argmax of score across all evaluated candidates
    if not all_candidates:
        # Fallback: return unary with zero evidence
        return ExpansionSelectionResult(
            selected_config=unary_config,
            selected_range=0,
            selected_order=1,
            range_results=[
                StepResult(
                    config=unary_config, num_features=0, score=0.0, rho=None,
                    stop_reason="exhaustive_argmax",
                ),
            ],
            order_results=[],
            phase1_stop_reason="exhaustive_argmax",
            phase2_stop_reason=None,
        )

    best = max(all_candidates, key=lambda s: s.score)
    best_config = best.config

    # Mark the best step with stop_reason
    for i, step in enumerate(range_results):
        if step.config == best_config:
            range_results[i] = step.model_copy(
                update={"stop_reason": "exhaustive_argmax"},
            )
    for i, step in enumerate(order_results):
        if step.config == best_config:
            order_results[i] = step.model_copy(
                update={"stop_reason": "exhaustive_argmax"},
            )

    return ExpansionSelectionResult(
        selected_config=best_config,
        selected_range=best_config.pairwise_max_range if best_config.max_order >= 2 else 0,
        selected_order=best_config.max_order,
        range_results=range_results,
        order_results=order_results,
        phase1_stop_reason="exhaustive_argmax",
        phase2_stop_reason=None,
    )
