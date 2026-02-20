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
    FeatureScaling,
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
        feature_config: FeatureExpansionConfig,
        prior_alpha: float = 1.0,
        prior_a0: float = 1e-6,
        prior_b0: float = 1e-6,
        feature_scaling: FeatureScaling = FeatureScaling.COUNTS_ONLY,
    ) -> None:
        self._seed = seed
        self._type_to_int, self._int_to_type = build_type_mapping(mixer_names)
        self._feature_config = feature_config
        self._prior_alpha = prior_alpha
        self._prior_a0 = prior_a0
        self._prior_b0 = prior_b0
        self._feature_scaling = feature_scaling

        # Fitted state
        self._mu_n: np.ndarray | None = None  # Posterior mean in scaled space (d,)
        self._mu_n_original: np.ndarray | None = None  # Posterior mean in original space (d,)
        self._L: np.ndarray | None = None  # Cholesky factor: LL^T = X^TX + αI
        self._X_mean: np.ndarray | None = None  # Feature means for centering
        self._diag_Vn: np.ndarray | None = None  # diag(V_n) in scaled space
        self._feature_scale: np.ndarray | None = None  # Per-feature divisor
        self._a_n: float = 0.0
        self._b_n: float = 0.0
        self._intercept: float = 0.0
        self._n_samples: int = 0
        self._n_features: int = 0

    def fit(self, assignments: list[tuple[str, ...]], y: np.ndarray) -> None:
        """Exact NIG conjugate update.

        Computes μ_n, V_n (via Cholesky of X^TX + αI), a_n, b_n.
        Stores diag(V_n) for diagonal variance decomposition.
        Stores full Cholesky for predict() and full-posterior LCB.
        """
        X = encode_batch(assignments, self._type_to_int, self._feature_config)
        n, d = X.shape
        self._n_samples = n
        self._n_features = d
        num_layers = len(assignments[0])

        # Center y to handle intercept (standard approach for Ridge)
        y_mean = float(np.mean(y))
        X_mean = np.mean(X, axis=0)
        self._X_mean = X_mean
        X_centered = X - X_mean
        y_centered = y - y_mean

        # Feature scaling: standardize after centering
        if self._feature_scaling is not FeatureScaling.NONE:
            feature_scale = self._build_feature_scale(X_centered, num_layers=num_layers)
            self._feature_scale = feature_scale
            X_work = X_centered / feature_scale
        else:
            self._feature_scale = None
            X_work = X_centered

        alpha = self._prior_alpha

        # A = X_work^T X_work + αI (in scaled space)
        A = X_work.T @ X_work + alpha * np.eye(d, dtype=np.float64)

        # Cholesky factorization: A = L L^T
        self._L = np.linalg.cholesky(A)

        # Posterior mean in scaled space: μ_n = A^{-1} X_work^T y_c
        Xty = X_work.T @ y_centered
        z = np.linalg.solve(self._L, Xty)  # L z = Xty
        self._mu_n = np.linalg.solve(self._L.T, z)  # L^T μ_n = z

        # Map back to original space for mean prediction + MRF
        if self._feature_scale is not None:
            self._mu_n_original = self._mu_n / self._feature_scale
        else:
            self._mu_n_original = self._mu_n

        # Intercept: y_mean - X_mean @ μ_n_original
        self._intercept = float(y_mean - X_mean @ self._mu_n_original)

        # Posterior hyperparameters for σ²
        self._a_n = self._prior_a0 + n / 2.0

        # b_n computed fully in centered+scaled space (no coordinate mixing)
        # SSE in scaled space: ||y_c - X_work @ μ_scaled||²
        fitted_residuals = y_centered - X_work @ self._mu_n
        SSE = float(fitted_residuals @ fitted_residuals)
        # Prior quadratic in scaled space: α * ||μ_scaled||²
        prior_quad = alpha * float(self._mu_n @ self._mu_n)
        self._b_n = max(self._prior_b0 + 0.5 * (SSE + prior_quad), 1e-12)

        # Diagonal of V_n = A^{-1} (in scaled space)
        L_inv = np.linalg.solve(self._L, np.eye(d, dtype=np.float64))
        self._diag_Vn = np.sum(L_inv ** 2, axis=0)

    def log_marginal_likelihood(self) -> float:
        """Log marginal likelihood of the data under the NIG model.

        Computed in scaled space (scaling is part of the model).
        All terms use quantities computed by fit().
        """
        assert self._L is not None, "Must call fit() first"
        from scipy.special import gammaln

        n, d = self._n_samples, self._n_features
        log_det_A = 2.0 * float(np.sum(np.log(np.diag(self._L))))
        return float(
            -0.5 * n * np.log(2.0 * np.pi)
            + 0.5 * d * np.log(self._prior_alpha)
            - 0.5 * log_det_A
            + self._prior_a0 * np.log(self._prior_b0)
            - self._a_n * np.log(self._b_n)
            + gammaln(self._a_n)
            - gammaln(self._prior_a0)
        )

    def _build_feature_scale(
        self, X_centered: np.ndarray, *, num_layers: int
    ) -> np.ndarray:
        """Build per-feature scale vector based on feature_scaling mode."""
        d = X_centered.shape[1]
        match self._feature_scaling:
            case FeatureScaling.NONE:
                return np.ones(d, dtype=np.float64)
            case FeatureScaling.COUNTS_ONLY:
                scale = np.ones(d, dtype=np.float64)
                fc = self._feature_config
                if fc.allocation_counts:
                    K = len(self._type_to_int)
                    scale[-K:] = float(num_layers)
                return scale
            case FeatureScaling.ALL:
                scale = np.std(X_centered, axis=0)
                scale[scale < 1e-12] = 1.0
                return scale

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
        # Mean prediction uses original-space weights
        mu = X @ self._mu_n_original + self._intercept

        # Compute variance via Cholesky solve (in scaled space)
        c = self._b_n / self._a_n
        X_centered = X - self._X_mean
        if self._feature_scale is not None:
            X_work = X_centered / self._feature_scale
        else:
            X_work = X_centered
        L_inv_X = np.linalg.solve(self._L, X_work.T)
        quad_forms = np.sum(L_inv_X ** 2, axis=0)
        sigma = np.sqrt(c * (1.0 + quad_forms))

        return mu, sigma

    def predict_mean(self, assignments: list[tuple[str, ...]]) -> np.ndarray:
        """Fast mean-only prediction.

        Returns:
            Predicted means, shape (n_samples,).
        """
        assert self._mu_n_original is not None, "Must call fit() first"
        X = encode_batch(assignments, self._type_to_int, self._feature_config)
        return X @ self._mu_n_original + self._intercept

    def predict_mean_one(self, assignment: tuple[str, ...] | list[str]) -> float:
        """Single-assignment mean prediction for SA hot loop.

        Uses a direct dot product — faster than going through batch encoding.
        """
        assert self._mu_n_original is not None, "Must call fit() first"
        x = encode_config(assignment, self._type_to_int, self._feature_config)
        return float(self._mu_n_original @ x + self._intercept)

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

        assert self._mu_n_original is not None, "Must call fit() first"
        sorted_names = tuple(sorted(self._type_to_int.keys()))
        return SurrogateMRFModel(
            coef=self._mu_n_original,
            intercept=self._intercept,
            num_layers=num_layers,
            mixer_names=sorted_names,
            feature_config=self._feature_config,
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

        assert self._mu_n_original is not None, "Must call fit() first"
        assert self._diag_Vn is not None

        sorted_names = tuple(sorted(self._type_to_int.keys()))

        # Create count-free feature config
        feature_config_no_counts = FeatureExpansionConfig(
            max_order=self._feature_config.max_order,
            pairwise_max_range=self._feature_config.pairwise_max_range,
            allocation_counts=False,
        )

        # Compute dimension without counts
        from place_layers.surrogate.encoding import compute_feature_dim

        dim_no_counts = compute_feature_dim(
            num_layers, len(sorted_names), feature_config_no_counts
        )

        # Mean model uses original-space weights
        mean_model = SurrogateMRFModel(
            coef=self._mu_n_original[:dim_no_counts],
            intercept=self._intercept,
            num_layers=num_layers,
            mixer_names=sorted_names,
            feature_config=feature_config_no_counts,
        )

        # Var model uses original-space diagonal variance
        if self._feature_scale is not None:
            diag_vn_original = self._diag_Vn / (self._feature_scale ** 2)
        else:
            diag_vn_original = self._diag_Vn
        var_model = SurrogateMRFModel(
            coef=diag_vn_original[:dim_no_counts],
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
    def degrees_of_freedom(self) -> float:
        """Degrees of freedom for the Student-t predictive: nu = 2*a_n."""
        assert self._a_n > 0, "Must call fit() first"
        return 2.0 * self._a_n

    @property
    def noise_scale_sq(self) -> float:
        """b_n / a_n — the posterior estimate of σ²."""
        assert self._b_n > 0, "Must call fit() first"
        return self._b_n / self._a_n

    @property
    def diag_Vn(self) -> np.ndarray:
        """Diagonal of the posterior covariance V_n (original space)."""
        assert self._diag_Vn is not None, "Must call fit() first"
        if self._feature_scale is not None:
            return self._diag_Vn / (self._feature_scale ** 2)
        return self._diag_Vn
