"""Random Forest surrogate model with uncertainty estimation.

Wraps sklearn's RandomForestRegressor to provide (mean, std) predictions
by computing variance across individual tree predictions.

Integer encoding of string-typed placements is encapsulated here — callers
pass placement tuples of mixer-name strings and never touch the ordinal
mapping.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from place_layers.surrogate.base_surrogate import BaseSurrogate
from place_layers.surrogate.encoding import (
    FeatureExpansionConfig,
    build_type_mapping,
    encode_batch,
    encode_config,
    feature_names as _feature_names,
)


class ForestSurrogate(BaseSurrogate):
    """Random Forest surrogate with predict-with-uncertainty.

    Not a Pydantic model — a plain wrapper whose parameters are passed
    from command config.
    """

    def __init__(
        self,
        n_estimators: int,
        max_depth: int | None,
        seed: int,
        mixer_names: list[str],
        feature_config: FeatureExpansionConfig,
        min_samples_leaf: int = 1,
    ) -> None:
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            n_jobs=-1,
        )
        self._seed = seed
        self._min_samples_leaf = min_samples_leaf
        self._is_fitted = False
        self._type_to_int, self._int_to_type = build_type_mapping(mixer_names)
        self._feature_config = feature_config

    def fit(self, assignments: Sequence[Sequence[str]], y: np.ndarray) -> None:
        """Fit the random forest on training data.

        Args:
            assignments: List of placements (each a tuple of mixer names).
            y: Target values of shape (n_samples,).
        """
        X = encode_batch(assignments, self._type_to_int, self._feature_config)
        self._model.fit(X, y)
        self._is_fitted = True

    def predict_mean(self, assignments: Sequence[Sequence[str]]) -> np.ndarray:
        """Fast mean-only prediction using sklearn's optimized predict."""
        X = encode_batch(assignments, self._type_to_int, self._feature_config)
        return self._model.predict(X)

    def predict_mean_one(self, assignment: Sequence[str]) -> float:
        """Single-assignment mean prediction for SA hot loop."""
        x = encode_config(
            assignment, self._type_to_int, self._feature_config
        ).reshape(1, -1)
        return float(self._model.predict(x)[0])

    def predict(
        self, assignments: Sequence[Sequence[str]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty from tree ensemble variance.

        Args:
            assignments: List of placements (each a tuple of mixer names).

        Returns:
            (mean, std) arrays, each of shape (n_samples,).
        """
        X = encode_batch(assignments, self._type_to_int, self._feature_config)
        # Collect predictions from each tree
        tree_preds = np.array(
            [tree.predict(X) for tree in self._model.estimators_]
        )
        mean = np.mean(tree_preds, axis=0)
        std = np.std(tree_preds, axis=0)
        return mean, std

    def feature_importances(self) -> np.ndarray:
        """Return feature importances from the fitted RF.

        Returns:
            Array of shape (n_features,).
        """
        return self._model.feature_importances_

    def feature_names(self, num_layers: int) -> list[str]:
        """Human-readable feature names for diagnostics."""
        return _feature_names(
            num_layers, self._int_to_type, self._feature_config
        )

    def cross_validate(
        self,
        assignments: list[tuple[str, ...]],
        y: np.ndarray,
        cv_folds: int,
    ) -> float:
        """Compute mean Spearman rank correlation via k-fold CV.

        Args:
            assignments: List of placements (each a tuple of mixer names).
            y: Target values of shape (n_samples,).
            cv_folds: Number of cross-validation folds.

        Returns:
            Mean Spearman rho across folds.
        """
        X = encode_batch(assignments, self._type_to_int, self._feature_config)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self._seed)
        rhos = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            rf = RandomForestRegressor(
                n_estimators=self._model.n_estimators,
                max_depth=self._model.max_depth,
                min_samples_leaf=self._min_samples_leaf,
                random_state=self._seed,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            preds = rf.predict(X_val)

            if len(y_val) < 2:
                continue

            rho, _ = spearmanr(y_val, preds)
            if np.isnan(rho):
                rho = 0.0
            rhos.append(rho)

        return float(np.mean(np.array(rhos))) if rhos else 0.0

    def diagnostics(
        self,
        assignments: Sequence[Sequence[str]],
        y: np.ndarray,
        *,
        folds: int | None,
        rng_seed: int = 0,
    ) -> "FrequentistCVDiagnostics":
        """Frequentist CV diagnostics: Spearman rank correlation."""
        from place_layers.surrogate.base_surrogate import FrequentistCVDiagnostics

        if folds is not None and len(y) >= folds * 2:
            rho = self.cross_validate(
                [tuple(a) for a in assignments], y, cv_folds=folds,
            )
        else:
            rho = 0.0
        return FrequentistCVDiagnostics(spearman_cv=rho)
