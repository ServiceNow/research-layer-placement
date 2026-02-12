"""Ridge + Bootstrap ensemble surrogate for placement optimization.

Uses sklearn's RidgeCV on expanded cluster-expansion features.  Uncertainty
is estimated via bootstrap resampling: *n_bootstraps* Ridge models are fit
on resampled data; std across their predictions gives a cheap sigma for
Expected Improvement acquisition.

Integer encoding of string-typed placements is encapsulated here — callers
pass placement tuples of mixer-name strings and never touch the ordinal
mapping.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold

from place_layers.surrogate.encoding import (
    FeatureExpansionConfig,
    build_type_mapping,
    encode_batch,
    encode_config,
)


class RidgeSurrogate:
    """Ridge regression surrogate with bootstrap uncertainty.

    Not a Pydantic model — a plain wrapper whose parameters are passed
    from command config.
    """

    def __init__(
        self,
        mixer_names: list[str],
        seed: int,
        n_bootstraps: int = 50,
        feature_config: FeatureExpansionConfig | None = None,
    ) -> None:
        self._seed = seed
        self._n_bootstraps = n_bootstraps
        self._type_to_int, self._int_to_type = build_type_mapping(mixer_names)
        self._feature_config = feature_config

        # Fitted state
        self._coef: np.ndarray | None = None
        self._intercept: float = 0.0
        self._main_model: RidgeCV | None = None
        self._bootstrap_models: list[Ridge] = []

    def fit(self, assignments: list[tuple[str, ...]], y: np.ndarray) -> None:
        """Fit main RidgeCV on full data + bootstrap Ridge models.

        Args:
            assignments: List of placements (each a tuple of mixer names).
            y: Target values of shape (n_samples,).
        """
        X = encode_batch(assignments, self._type_to_int, self._feature_config)
        rng = np.random.RandomState(self._seed)

        # Main model: RidgeCV selects best alpha automatically
        main = RidgeCV(alphas=np.logspace(-3, 3, 7), gcv_mode="svd")
        main.fit(X, y)
        self._main_model = main
        self._coef = main.coef_
        self._intercept = float(main.intercept_)

        # Bootstrap ensemble for uncertainty
        n = len(y)
        self._bootstrap_models = []
        for _ in range(self._n_bootstraps):
            idx = rng.choice(n, size=n, replace=True)
            model = Ridge(alpha=main.alpha_, solver="svd")
            model.fit(X[idx], y[idx])
            self._bootstrap_models.append(model)

    def predict_mean(self, assignments: list[tuple[str, ...]]) -> np.ndarray:
        """Fast mean-only prediction using main Ridge model.

        Args:
            assignments: List of placements.

        Returns:
            Predicted means, shape (n_samples,).
        """
        assert self._main_model is not None, "Must call fit() first"
        X = encode_batch(assignments, self._type_to_int, self._feature_config)
        return self._main_model.predict(X)

    def predict_mean_one(self, assignment: tuple[str, ...] | list[str]) -> float:
        """Single-assignment mean prediction for SA hot loop.

        Uses a direct dot product — faster than going through sklearn predict.
        """
        assert self._coef is not None, "Must call fit() first"
        x = encode_config(
            assignment, self._type_to_int, self._feature_config
        )
        return float(self._coef @ x + self._intercept)

    def predict(
        self, assignments: list[tuple[str, ...]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty from bootstrap ensemble.

        Args:
            assignments: List of placements.

        Returns:
            (mean, std) arrays, each of shape (n_samples,).
        """
        assert self._main_model is not None, "Must call fit() first"
        X = encode_batch(assignments, self._type_to_int, self._feature_config)

        # Collect predictions from each bootstrap model
        boot_preds = np.array([m.predict(X) for m in self._bootstrap_models])
        mean = np.mean(boot_preds, axis=0)
        std = np.std(boot_preds, axis=0)
        return mean, std

    def cross_validate(
        self,
        assignments: list[tuple[str, ...]],
        y: np.ndarray,
        cv_folds: int,
    ) -> float:
        """Compute mean Spearman rank correlation via k-fold CV.

        Args:
            assignments: List of placements.
            y: Target values of shape (n_samples,).
            cv_folds: Number of cross-validation folds.

        Returns:
            Mean Spearman rho across folds.
        """
        X = encode_batch(assignments, self._type_to_int, self._feature_config)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self._seed)
        rhos: list[float] = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            ridge = RidgeCV(alphas=np.logspace(-3, 3, 7), gcv_mode="svd")
            ridge.fit(X_train, y_train)
            preds = ridge.predict(X_val)

            if len(y_val) < 2:
                continue

            rho, _ = spearmanr(y_val, preds)
            if np.isnan(rho):
                rho = 0.0
            rhos.append(rho)

        return float(np.mean(np.array(rhos))) if rhos else 0.0
