"""Bayesian Ridge surrogate with exact Normal-Inverse-Gamma posterior.

Replaces bootstrap-based uncertainty with closed-form Bayesian posterior
variance.  The conjugate NIG model gives:

- Exact posterior mean: μ_n = V_n (X^T y)  where V_n = (X^T X + αI)^{-1}
- Predictive variance decomposable into per-feature diagonal terms
- Full-posterior LCB for post-hoc candidate ranking

This avoids the "frontier above cloud" problem where bootstrap uncertainty
doesn't grow in extrapolation regions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

if TYPE_CHECKING:
    from place_layers.mrf.surrogate_model import SurrogateMRFModel

from place_layers.surrogate.encoding import (
    FeatureExpansionConfig,
    build_type_mapping,
    encode_batch,
    encode_config,
)


class BayesianRidgeSurrogate:
    """Bayesian Ridge surrogate with Normal-Inverse-Gamma conjugate posterior.

    Prior: w | σ² ~ N(0, σ²·α⁻¹I), σ² ~ InvGamma(a₀, b₀).
    Posterior: w | σ², D ~ N(μ_n, σ²·V_n).
    Predictive: Student-t with decomposable diagonal variance.

    Same public interface as RidgeSurrogate.
    """

    def __init__(
        self,
        mixer_names: list[str],
        seed: int,
        feature_config: FeatureExpansionConfig | None = None,
        prior_alpha: float = 1.0,
        prior_a0: float = 1e-6,
        prior_b0: float = 1e-6,
    ) -> None:
        self._seed = seed
        self._type_to_int, self._int_to_type = build_type_mapping(mixer_names)
        self._feature_config = feature_config
        self._prior_alpha = prior_alpha
        self._prior_a0 = prior_a0
        self._prior_b0 = prior_b0

        # Fitted state
        self._mu_n: np.ndarray | None = None  # Posterior mean (d,)
        self._L: np.ndarray | None = None  # Cholesky factor: LL^T = X^TX + αI
        self._diag_Vn: np.ndarray | None = None  # diag(V_n) for decomposition
        self._a_n: float = 0.0
        self._b_n: float = 0.0
        self._intercept: float = 0.0

    def fit(self, assignments: list[tuple[str, ...]], y: np.ndarray) -> None:
        """Exact NIG conjugate update.

        Computes μ_n, V_n (via Cholesky of X^TX + αI), a_n, b_n.
        Stores diag(V_n) for diagonal variance decomposition.
        Stores full Cholesky for predict() and full-posterior LCB.
        """
        X = encode_batch(assignments, self._type_to_int, self._feature_config)
        n, d = X.shape

        # Center y to handle intercept (standard approach for Ridge)
        y_mean = float(np.mean(y))
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean
        y_centered = y - y_mean

        alpha = self._prior_alpha

        # A = X_c^T X_c + αI
        A = X_centered.T @ X_centered + alpha * np.eye(d, dtype=np.float64)

        # Cholesky factorization: A = L L^T
        self._L = np.linalg.cholesky(A)

        # Posterior mean: μ_n = V_n X_c^T y_c = A^{-1} X_c^T y_c
        # Solve via Cholesky: A μ_n = X_c^T y_c
        Xty = X_centered.T @ y_centered
        z = np.linalg.solve(self._L, Xty)  # L z = Xty
        self._mu_n = np.linalg.solve(self._L.T, z)  # L^T μ_n = z

        # Intercept: y_mean - X_mean @ μ_n
        self._intercept = float(y_mean - X_mean @ self._mu_n)

        # Posterior hyperparameters for σ²
        self._a_n = self._prior_a0 + n / 2.0

        # Safe b_n computation: b_n = b_0 + 0.5*(y_c^T y_c - μ_n^T A μ_n)
        # μ_n^T A μ_n = ||L^T μ_n||² (no explicit inverse needed)
        Lt_mu = self._L.T @ self._mu_n
        self._b_n = self._prior_b0 + 0.5 * (
            float(y_centered @ y_centered) - float(Lt_mu @ Lt_mu)
        )
        # Ensure non-negative (numerical safety)
        self._b_n = max(self._b_n, 1e-12)

        # Diagonal of V_n = A^{-1}: solve A V = I column by column is expensive.
        # Instead: V_n = L^{-T} L^{-1}, so diag(V_n) = sum of squares of rows of L^{-1}
        # L_inv = inv(L), then diag(V_n) = (L_inv**2).sum(axis=0)
        L_inv = np.linalg.solve(self._L, np.eye(d, dtype=np.float64))
        self._diag_Vn = np.sum(L_inv ** 2, axis=0)

    def predict(
        self, assignments: list[tuple[str, ...]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Full Student-t predictive: mean and scale.

        Uses full V_n (not diagonal) for accurate uncertainty.
        mu = X @ μ_n + intercept
        sigma = sqrt(c · (1 + ||L^{-1} x_c||²)) for each x
        where c = b_n / a_n and x_c = x - X_mean (centered features).

        Returns:
            (mean, sigma) arrays, each of shape (n_samples,).
        """
        assert self._mu_n is not None, "Must call fit() first"
        X = encode_batch(assignments, self._type_to_int, self._feature_config)
        mu = X @ self._mu_n + self._intercept

        # Compute variance via Cholesky solve
        c = self._b_n / self._a_n
        # Solve L z = X^T for each sample (X rows)
        # L_inv_X = L^{-1} X^T, shape (d, n)
        L_inv_X = np.linalg.solve(self._L, X.T)
        # Variance for each sample: c * (1 + ||L^{-1} x_i||²)
        quad_forms = np.sum(L_inv_X ** 2, axis=0)
        sigma = np.sqrt(c * (1.0 + quad_forms))

        return mu, sigma

    def predict_mean(self, assignments: list[tuple[str, ...]]) -> np.ndarray:
        """Fast mean-only prediction.

        Returns:
            Predicted means, shape (n_samples,).
        """
        assert self._mu_n is not None, "Must call fit() first"
        X = encode_batch(assignments, self._type_to_int, self._feature_config)
        return X @ self._mu_n + self._intercept

    def predict_mean_one(self, assignment: tuple[str, ...] | list[str]) -> float:
        """Single-assignment mean prediction for SA hot loop.

        Uses a direct dot product — faster than going through batch encoding.
        """
        assert self._mu_n is not None, "Must call fit() first"
        x = encode_config(assignment, self._type_to_int, self._feature_config)
        return float(self._mu_n @ x + self._intercept)

    def cross_validate(
        self,
        assignments: list[tuple[str, ...]],
        y: np.ndarray,
        cv_folds: int,
    ) -> float:
        """Compute mean Spearman rank correlation via k-fold CV.

        Uses RidgeCV (same as RidgeSurrogate) for fold-level fitting.

        Returns:
            Mean Spearman rho across folds.
        """
        X = encode_batch(assignments, self._type_to_int, self._feature_config)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self._seed)
        rhos: list[float] = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            ridge = RidgeCV(alphas=np.logspace(-3, 3, 7))
            ridge.fit(X_train, y_train)
            preds = ridge.predict(X_val)

            if len(y_val) < 2:
                continue

            rho, _ = spearmanr(y_val, preds)
            if np.isnan(rho):
                rho = 0.0
            rhos.append(rho)

        return float(np.mean(np.array(rhos))) if rhos else 0.0

    def extract_mrf_model(self, num_layers: int) -> "SurrogateMRFModel":
        """Extract MRF model from MAP weights μ_n (same as RidgeSurrogate)."""
        from place_layers.mrf.surrogate_model import SurrogateMRFModel

        assert self._mu_n is not None, "Must call fit() first"
        sorted_names = tuple(sorted(self._type_to_int.keys()))
        feature_config = self._feature_config or FeatureExpansionConfig()
        return SurrogateMRFModel(
            coef=self._mu_n,
            intercept=self._intercept,
            num_layers=num_layers,
            mixer_names=sorted_names,
            feature_config=feature_config,
        )

    def extract_mean_var_mrf_models(
        self, num_layers: int
    ) -> tuple["SurrogateMRFModel", "SurrogateMRFModel"]:
        """Extract (mean_model, var_model) for ParetoLCBSolver.

        Both models exclude count features:
        - mean_coef = μ_n[:dim_no_counts]
        - var_coef = diag(V_n)[:dim_no_counts]
        - Both built with feature_config that has allocation_counts=False
        """
        from place_layers.mrf.surrogate_model import SurrogateMRFModel

        assert self._mu_n is not None, "Must call fit() first"
        assert self._diag_Vn is not None

        sorted_names = tuple(sorted(self._type_to_int.keys()))
        feature_config = self._feature_config or FeatureExpansionConfig()

        # Create count-free feature config
        feature_config_no_counts = FeatureExpansionConfig(
            max_order=feature_config.max_order,
            pairwise_max_range=feature_config.pairwise_max_range,
            allocation_counts=False,
        )

        # Compute dimension without counts
        from place_layers.surrogate.encoding import compute_feature_dim

        dim_no_counts = compute_feature_dim(
            num_layers, len(sorted_names), feature_config_no_counts
        )

        # Mean model: μ_n[:dim_no_counts]
        mean_model = SurrogateMRFModel(
            coef=self._mu_n[:dim_no_counts],
            intercept=self._intercept,
            num_layers=num_layers,
            mixer_names=sorted_names,
            feature_config=feature_config_no_counts,
        )

        # Var model: diag(V_n)[:dim_no_counts]
        var_model = SurrogateMRFModel(
            coef=self._diag_Vn[:dim_no_counts],
            intercept=0.0,
            num_layers=num_layers,
            mixer_names=sorted_names,
            feature_config=feature_config_no_counts,
        )

        return mean_model, var_model

    def full_lcb(
        self, assignments: list[tuple[str, ...]], beta: float
    ) -> np.ndarray:
        """Full-posterior LCB using ALL features (including counts).

        score = m(x) - β·sqrt(c·(1 + x^T V_n x))

        Used for post-hoc candidate ranking after DP.

        Returns:
            LCB scores, shape (n_candidates,).
        """
        assert self._mu_n is not None, "Must call fit() first"
        mu, sigma = self.predict(assignments)
        return mu - beta * sigma

    @property
    def noise_scale_sq(self) -> float:
        """b_n / a_n — the posterior estimate of σ²."""
        assert self._b_n > 0, "Must call fit() first"
        return self._b_n / self._a_n

    @property
    def diag_Vn(self) -> np.ndarray:
        """Diagonal of the posterior covariance V_n."""
        assert self._diag_Vn is not None, "Must call fit() first"
        return self._diag_Vn
