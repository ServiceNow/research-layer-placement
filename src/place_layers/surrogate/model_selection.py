"""RF model selection via regularization grid search.

Automated hyperparameter tuning for the Random Forest surrogate.  At each
exploration checkpoint the grid is swept with k-fold CV; the best
configuration (highest test Spearman rho, tie-break by smallest gap) is
persisted in the exploration dataset and later consumed by the surrogate
search.

Also provides automatic expansion level selection via RidgeCV to determine
the optimal cluster expansion depth (unary, pairwise, triplets, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from place_layers.surrogate.encoding import (
    FeatureExpansionConfig,
    build_type_mapping,
    compute_feature_dim,
    encode_batch,
)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class RFParams(BaseModel):
    """Frozen container for Random Forest hyperparameters."""

    model_config = {"frozen": True}

    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int


class ModelSelectionResult(BaseModel):
    """Result of automated RF model selection."""

    model_config = {"frozen": True}

    best_params: RFParams
    test_rho: float
    train_rho: float
    gap: float
    top_region_rho: float | None


class LevelResult(BaseModel):
    """Per-level diagnostics from expansion selection."""

    model_config = {"frozen": True}

    level: int
    config: FeatureExpansionConfig
    num_features: int
    rho: float


class ExpansionSelectionResult(BaseModel):
    """Result of automatic expansion order selection."""

    model_config = {"frozen": True}

    selected_config: FeatureExpansionConfig
    selected_level: int
    level_results: list[LevelResult]


@dataclass(frozen=True)
class _GridCVResult:
    """Internal: per-grid-point CV summary."""

    max_depth: int | None
    min_samples_leaf: int
    train_rho: float
    test_rho: float
    gap: float


# ---------------------------------------------------------------------------
# Regularization grid
# ---------------------------------------------------------------------------

REGULARIZATION_GRID: list[dict[str, int | None]] = [
    {"max_depth": None, "min_samples_leaf": 1},     # max capacity
    {"max_depth": None, "min_samples_leaf": 3},
    {"max_depth": None, "min_samples_leaf": 5},
    {"max_depth": None, "min_samples_leaf": 10},
    {"max_depth": None, "min_samples_leaf": 20},
    {"max_depth": None, "min_samples_leaf": 50},
    {"max_depth": 12,   "min_samples_leaf": 3},
    {"max_depth": 8,    "min_samples_leaf": 3},
    {"max_depth": 6,    "min_samples_leaf": 3},
    {"max_depth": 6,    "min_samples_leaf": 10},
]


# ---------------------------------------------------------------------------
# Expansion level sequence
# ---------------------------------------------------------------------------


def _build_level_sequence(max_order: int) -> list[FeatureExpansionConfig]:
    """Build the sequence of expansion configs to test.

    Level 0: unary only (max_order=1)
    Level 1: + adjacent pairs (max_order=2, range=1)
    Level 2: + skip-1 pairs (max_order=2, range=2)
    Level 3: + skip-2 pairs (max_order=2, range=3)
    Level 4: + triplets (max_order=3, range=3)
    Level 5: + quadruplets (max_order=4, range=3)
    ...
    """
    levels: list[FeatureExpansionConfig] = [
        FeatureExpansionConfig(max_order=1),
    ]
    if max_order < 2:
        return levels

    # Pairwise levels: range 1, 2, 3
    max_range = 3
    for r in range(1, max_range + 1):
        levels.append(FeatureExpansionConfig(max_order=2, pairwise_max_range=r))

    # Higher-order levels: 3, 4, ..., max_order
    for n in range(3, max_order + 1):
        levels.append(
            FeatureExpansionConfig(max_order=n, pairwise_max_range=max_range)
        )

    return levels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cv_spearman(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    cv_folds: int,
    seed: int,
) -> _GridCVResult:
    """Run k-fold CV for one grid point, return train/test Spearman rho."""
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    train_rhos: list[float] = []
    test_rhos: list[float] = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        # Test fold
        if len(y_val) >= 2:
            preds_val = rf.predict(X_val)
            rho_val, _ = spearmanr(y_val, preds_val)
            if np.isnan(rho_val):
                rho_val = 0.0
            test_rhos.append(rho_val)

        # Train fold
        if len(y_train) >= 2:
            preds_train = rf.predict(X_train)
            rho_train, _ = spearmanr(y_train, preds_train)
            if np.isnan(rho_train):
                rho_train = 0.0
            train_rhos.append(rho_train)

    mean_test = float(np.mean(test_rhos)) if test_rhos else 0.0
    mean_train = float(np.mean(train_rhos)) if train_rhos else 0.0

    return _GridCVResult(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        train_rho=mean_train,
        test_rho=mean_test,
        gap=mean_train - mean_test,
    )


def _compute_top_region_rho(
    X: np.ndarray,
    y: np.ndarray,
    params: RFParams,
    seed: int,
    top_fraction: float,
) -> float | None:
    """Fit on all data, compute Spearman rho among the top fraction by real score.

    Returns None if the top region has fewer than 2 samples.
    """
    n_top = max(1, int(len(y) * top_fraction))
    if n_top < 2:
        return None

    rf = RandomForestRegressor(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        min_samples_leaf=params.min_samples_leaf,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X, y)
    preds = rf.predict(X)

    # Select indices of top real scores
    top_indices = np.argsort(y)[-n_top:]
    rho, _ = spearmanr(y[top_indices], preds[top_indices])
    if np.isnan(rho):
        return 0.0
    return float(rho)


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

        ridge = RidgeCV(alphas=np.logspace(-3, 3, 7), gcv_mode="svd")
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


def select_rf_params(
    assignments: list[tuple[str, ...]],
    y: np.ndarray,
    n_estimators: int,
    cv_folds: int,
    seed: int,
    mixer_names: list[str],
    top_fraction: float = 0.05,
    max_gap: float | None = None,
    feature_config: FeatureExpansionConfig | None = None,
) -> ModelSelectionResult:
    """Select best RF hyperparameters via grid search over regularization configs.

    For each point in :data:`REGULARIZATION_GRID`, runs k-fold CV computing
    mean train and test Spearman rho.  Selects the configuration with the
    highest test rho (tie-break: smallest gap).

    When *max_gap* is set, configs with ``gap <= max_gap`` are preferred.
    If no config meets the gap constraint, all configs are considered as
    a fallback.

    After selection, fits the best config on all data and computes
    ``top_region_rho`` among the top *top_fraction* samples by real score.

    Args:
        assignments: Evaluated placements.
        y: Corresponding scores, shape ``(n,)``.
        n_estimators: Number of trees for all grid points.
        cv_folds: Number of CV folds.
        seed: Random seed.
        mixer_names: Mixer type names for encoding.
        top_fraction: Fraction of top samples for ``top_region_rho``.
        max_gap: If set, prefer configs with gap <= this value.
        feature_config: Optional feature expansion config for encoding.

    Returns:
        :class:`ModelSelectionResult` with the best params and diagnostics.
    """
    type_to_int, _ = build_type_mapping(mixer_names)
    X = encode_batch(assignments, type_to_int, feature_config)

    # Sweep grid
    results: list[_GridCVResult] = []
    for point in REGULARIZATION_GRID:
        res = _cv_spearman(
            X=X,
            y=y,
            n_estimators=n_estimators,
            max_depth=point["max_depth"],
            min_samples_leaf=point["min_samples_leaf"],
            cv_folds=cv_folds,
            seed=seed,
        )
        results.append(res)

    # Select: prefer configs within gap budget, fall back to all
    if max_gap is not None:
        within_gap = [r for r in results if r.gap <= max_gap]
        pool = within_gap if within_gap else results
    else:
        pool = results
    best = max(pool, key=lambda r: (r.test_rho, -r.gap))

    best_params = RFParams(
        n_estimators=n_estimators,
        max_depth=best.max_depth,
        min_samples_leaf=best.min_samples_leaf,
    )

    # Top-region rho
    top_rho = _compute_top_region_rho(X, y, best_params, seed, top_fraction)

    return ModelSelectionResult(
        best_params=best_params,
        test_rho=best.test_rho,
        train_rho=best.train_rho,
        gap=best.gap,
        top_region_rho=top_rho,
    )


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
    max_order: int = 3,
    seed: int = 42,
) -> ExpansionSelectionResult:
    """Automatic expansion order selection using RidgeCV.

    Tests increasing expansion levels (unary -> pairwise -> triplets -> ...)
    and selects the lowest level where either:
    - rho >= target_rho, OR
    - marginal improvement over previous level < min_improvement

    Args:
        assignments: Evaluated placements.
        scores: Corresponding scores, shape ``(n,)``.
        mixer_names: Mixer type names for encoding.
        cv_folds: Number of CV folds.
        target_rho: Target Spearman rho for early termination.
        min_improvement: Minimum rho improvement to keep climbing.
        max_order: Maximum n-body order to test.
        seed: Random seed.

    Returns:
        :class:`ExpansionSelectionResult` with selected config and per-level diagnostics.
    """
    type_to_int, _ = build_type_mapping(mixer_names)
    num_layers = len(assignments[0]) if assignments else 0
    num_types = len(type_to_int)

    level_sequence = _build_level_sequence(max_order)
    level_results: list[LevelResult] = []
    prev_rho = -1.0
    selected_level = 0

    for level_idx, config in enumerate(level_sequence):
        num_features = compute_feature_dim(num_layers, num_types, config)
        X = encode_batch(assignments, type_to_int, config)
        rho = _ridge_cv_spearman(X, scores, cv_folds, seed)

        level_results.append(
            LevelResult(
                level=level_idx,
                config=config,
                num_features=num_features,
                rho=rho,
            )
        )

        selected_level = level_idx

        # Check if target reached
        if rho >= target_rho:
            break

        # Check if marginal improvement is too small (only after first level)
        if level_idx > 0 and (rho - prev_rho) < min_improvement:
            # Select previous level if this one didn't improve enough
            selected_level = level_idx - 1
            break

        prev_rho = rho

    selected_config = level_results[selected_level].config

    return ExpansionSelectionResult(
        selected_config=selected_config,
        selected_level=selected_level,
        level_results=level_results,
    )
