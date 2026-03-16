"""Surrogate model configuration — discriminated union.

Provides ``SurrogateConfig``, a Pydantic discriminated union of
``RidgeSurrogateConfig`` and ``BayesianRidgeSurrogateConfig``.

Each config variant carries a ``create_surrogate()`` smart constructor
that returns the corresponding surrogate instance, so callers never need
to branch on ``surrogate_type`` to instantiate the model.

Following Hydra-pydantic best practices, no ``Field()`` carries a
``default=`` — all defaults live in the YAML config files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, Field

from place_layers.surrogate.encoding import FeatureExpansionConfig, FeatureScaling

if TYPE_CHECKING:
    from place_layers.surrogate.base_surrogate import BaseSurrogate


class BaseSurrogateConfig(BaseModel):
    """Base for all surrogate model configs."""

    surrogate_type: str = Field(description="Surrogate model discriminator.")

    def create_surrogate(
        self,
        mixer_names: list[str],
        seed: int,
        feature_config: FeatureExpansionConfig,
    ) -> "BaseSurrogate":
        """Smart constructor — subclasses override."""
        raise NotImplementedError


class RidgeSurrogateConfig(BaseSurrogateConfig):
    """Ridge + Bootstrap ensemble surrogate configuration."""

    surrogate_type: Literal["ridge"] = "ridge"

    n_bootstraps: int = Field(
        description="Number of bootstrap models for uncertainty.",
        gt=0,
    )
    ei_xi: float = Field(
        description="Exploration parameter for Expected Improvement.",
        ge=0,
    )
    cv_folds: int = Field(
        description="K-fold CV folds for surrogate diagnostics (Spearman rho).",
        gt=1,
    )
    target_rank_correlation: float = Field(
        description="Stop exploration when Ridge CV Spearman rho reaches this.",
        gt=0,
        le=1,
    )

    def create_surrogate(
        self,
        mixer_names: list[str],
        seed: int,
        feature_config: FeatureExpansionConfig,
    ) -> "BaseSurrogate":
        from place_layers.surrogate.ridge_surrogate import RidgeSurrogate

        return RidgeSurrogate(
            mixer_names=mixer_names,
            seed=seed,
            n_bootstraps=self.n_bootstraps,
            feature_config=feature_config,
        )


class BayesianRidgeSurrogateConfig(BaseSurrogateConfig):
    """Bayesian Ridge surrogate configuration (NIG conjugate posterior)."""

    surrogate_type: Literal["bayesian_ridge"] = "bayesian_ridge"

    prior_alpha: float = Field(
        description="Ridge regularization / prior precision.",
        gt=0,
    )
    alpha_selection: Literal["fixed", "evidence"] = Field(
        description="Alpha selection mode: 'fixed' uses prior_alpha as-is, "
        "'evidence' optimizes alpha via Type-II ML (grid search on log "
        "marginal likelihood) on the initial fit.",
    )
    prior_a0: float = Field(
        description="InvGamma shape for noise variance prior.",
        gt=0,
    )
    prior_b0: float = Field(
        description="InvGamma scale for noise variance prior.",
        gt=0,
    )
    feature_scaling: FeatureScaling = Field(
        description="Feature scaling mode: none, counts_only, or all.",
    )

    def create_surrogate(
        self,
        mixer_names: list[str],
        seed: int,
        feature_config: FeatureExpansionConfig,
    ) -> "BaseSurrogate":
        from place_layers.surrogate.bayesian_ridge import BayesianRidgeSurrogate

        return BayesianRidgeSurrogate(
            mixer_names=mixer_names,
            seed=seed,
            feature_config=feature_config,
            prior_alpha=self.prior_alpha,
            prior_a0=self.prior_a0,
            prior_b0=self.prior_b0,
            feature_scaling=self.feature_scaling,
        )


SurrogateConfig = Annotated[
    RidgeSurrogateConfig | BayesianRidgeSurrogateConfig,
    Field(discriminator="surrogate_type"),
]
