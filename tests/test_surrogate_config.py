"""Tests for surrogate config discriminated union and smart constructors."""

import pytest
from pydantic import TypeAdapter, ValidationError

from place_layers.surrogate.config import (
    BayesianRidgeSurrogateConfig,
    RidgeSurrogateConfig,
    SurrogateConfig,
)
from place_layers.surrogate.bayesian_ridge import BayesianRidgeSurrogate
from place_layers.surrogate.encoding import FeatureExpansionConfig
from place_layers.surrogate.ridge_surrogate import RidgeSurrogate


_MIXER_NAMES = ["m0", "m1"]


class TestRidgeSurrogateConfig:
    """Tests for RidgeSurrogateConfig."""

    def test_create_surrogate(self) -> None:
        config = RidgeSurrogateConfig(n_bootstraps=30, ei_xi=0.02, cv_folds=5, target_rank_correlation=0.95)
        feature_config = FeatureExpansionConfig(max_order=1, pairwise_max_range=1, allocation_counts=True)
        surrogate = config.create_surrogate(mixer_names=_MIXER_NAMES, seed=42, feature_config=feature_config)
        assert isinstance(surrogate, RidgeSurrogate)
        assert surrogate._n_bootstraps == 30

    def test_fields_required(self) -> None:
        with pytest.raises(ValidationError):
            RidgeSurrogateConfig()  # missing n_bootstraps, ei_xi

    def test_validation_constraints(self) -> None:
        with pytest.raises(ValidationError):
            RidgeSurrogateConfig(n_bootstraps=0, ei_xi=0.01, cv_folds=5, target_rank_correlation=0.95)  # gt=0
        with pytest.raises(ValidationError):
            RidgeSurrogateConfig(n_bootstraps=50, ei_xi=-1.0, cv_folds=5, target_rank_correlation=0.95)  # ge=0


class TestBayesianRidgeSurrogateConfig:
    """Tests for BayesianRidgeSurrogateConfig."""

    def test_create_surrogate(self) -> None:
        config = BayesianRidgeSurrogateConfig(
            prior_alpha=2.0, prior_a0=1e-4, prior_b0=1e-4,
            feature_scaling="none", alpha_selection="fixed",
        )
        feature_config = FeatureExpansionConfig(max_order=1, pairwise_max_range=1, allocation_counts=True)
        surrogate = config.create_surrogate(mixer_names=_MIXER_NAMES, seed=42, feature_config=feature_config)
        assert isinstance(surrogate, BayesianRidgeSurrogate)
        assert surrogate._prior_alpha == 2.0

    def test_fields_required(self) -> None:
        with pytest.raises(ValidationError):
            BayesianRidgeSurrogateConfig()  # missing required fields

    def test_validation_constraints(self) -> None:
        with pytest.raises(ValidationError):
            BayesianRidgeSurrogateConfig(
                prior_alpha=0.0, prior_a0=1e-6, prior_b0=1e-6,
                feature_scaling="none", alpha_selection="fixed",
            )  # prior_alpha gt=0


class TestSurrogateConfigUnion:
    """Tests for SurrogateConfig discriminated union parsing."""

    _adapter = TypeAdapter(SurrogateConfig)

    def test_parse_ridge(self) -> None:
        config = self._adapter.validate_python(
            {"surrogate_type": "ridge", "n_bootstraps": 50, "ei_xi": 0.01, "cv_folds": 5, "target_rank_correlation": 0.95}
        )
        assert isinstance(config, RidgeSurrogateConfig)
        assert config.n_bootstraps == 50

    def test_parse_bayesian_ridge(self) -> None:
        config = self._adapter.validate_python(
            {
                "surrogate_type": "bayesian_ridge",
                "prior_alpha": 1.0,
                "prior_a0": 1e-6,
                "prior_b0": 1e-6,
                "feature_scaling": "none",
                "alpha_selection": "fixed",
            }
        )
        assert isinstance(config, BayesianRidgeSurrogateConfig)
        assert config.prior_alpha == 1.0

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._adapter.validate_python(
                {"surrogate_type": "unknown", "n_bootstraps": 50, "ei_xi": 0.01}
            )

    def test_mismatched_fields_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._adapter.validate_python(
                {"surrogate_type": "ridge", "prior_alpha": 1.0, "cv_folds": 5}
            )
