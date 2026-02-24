"""Tests for TraceLikelihood normalization feature."""

import pytest

from place_layers.evaluator.trace_likelihood import (
    LinearNormalization,
    NoNormalization,
)


class TestNoNormalization:
    """Test NoNormalization (identity function)."""

    def test_returns_unchanged_values(self):
        """NoNormalization should return values unchanged."""
        norm = NoNormalization()

        assert norm.normalize(-1000) == -1000
        assert norm.normalize(-100) == -100
        assert norm.normalize(0) == 0
        assert norm.normalize(100) == 100

    def test_type_field(self):
        """NoNormalization should have type='none'."""
        norm = NoNormalization()
        assert norm.type == "none"


class TestLinearNormalization:
    """Test LinearNormalization (map [low, high] to [0, 1])."""

    def test_maps_bounds_correctly(self):
        """LinearNormalization should map low→0 and high→1."""
        norm = LinearNormalization(low=-1000, high=-100)

        assert norm.normalize(-1000) == 0.0
        assert norm.normalize(-100) == 1.0

    def test_maps_midpoint_correctly(self):
        """LinearNormalization should map midpoint to 0.5."""
        norm = LinearNormalization(low=-1000, high=-100)

        # Midpoint: (-1000 + -100) / 2 = -550
        assert norm.normalize(-550) == 0.5

    def test_handles_values_outside_range(self):
        """LinearNormalization should extrapolate for values outside [low, high]."""
        norm = LinearNormalization(low=-1000, high=-100)

        # Below low → negative
        assert norm.normalize(-1500) < 0.0

        # Above high → greater than 1
        assert norm.normalize(-50) > 1.0

    def test_handles_equal_bounds(self):
        """LinearNormalization should return 0.5 when low == high (avoid division by zero)."""
        norm = LinearNormalization(low=-500, high=-500)

        assert norm.normalize(-500) == 0.5
        assert norm.normalize(-1000) == 0.5
        assert norm.normalize(0) == 0.5

    def test_positive_range(self):
        """LinearNormalization should work with positive ranges."""
        norm = LinearNormalization(low=0, high=100)

        assert norm.normalize(0) == 0.0
        assert norm.normalize(100) == 1.0
        assert norm.normalize(50) == 0.5

    def test_inverted_range(self):
        """LinearNormalization should work when low > high (inverted scale)."""
        norm = LinearNormalization(low=-100, high=-1000)

        # With inverted range, -100 still maps to 0, -1000 to 1
        # But the mapping is inverted: higher values → lower normalized values
        assert norm.normalize(-100) == 0.0
        assert norm.normalize(-1000) == 1.0
        assert norm.normalize(-550) == 0.5

    def test_type_field(self):
        """LinearNormalization should have type='linear'."""
        norm = LinearNormalization(low=-1000, high=-100)
        assert norm.type == "linear"

    def test_rejects_infinite_bounds(self):
        """LinearNormalization should reject infinite bounds."""
        with pytest.raises(ValueError, match="must be finite"):
            LinearNormalization(low=float("-inf"), high=-100)

        with pytest.raises(ValueError, match="must be finite"):
            LinearNormalization(low=-1000, high=float("inf"))


class TestNormalizationConfig:
    """Test that normalization configs work with Pydantic."""

    def test_discriminated_union(self):
        """Test that type field discriminates between normalization types."""
        from pydantic import TypeAdapter
        from place_layers.evaluator.trace_likelihood import Normalization

        adapter = TypeAdapter(Normalization)

        # Test NoNormalization
        no_norm = adapter.validate_python({"type": "none"})
        assert isinstance(no_norm, NoNormalization)

        # Test LinearNormalization
        lin_norm = adapter.validate_python({
            "type": "linear",
            "low": -1000,
            "high": -100,
        })
        assert isinstance(lin_norm, LinearNormalization)
        assert lin_norm.low == -1000
        assert lin_norm.high == -100

    def test_default_is_no_normalization(self):
        """Test that default normalization is NoNormalization."""
        from place_layers.evaluator.trace_likelihood import TraceLikelihoodEvaluatorConfig
        from place_layers.backend.mock import MockBackendConfig, ReplayConfig

        # Create minimal config (normalization not specified)
        # Note: We can't instantiate without proper backend config, so just check the field default
        field_info = TraceLikelihoodEvaluatorConfig.model_fields["normalization"]
        default = field_info.default

        # The default should be NoNormalization
        assert isinstance(default, NoNormalization)
