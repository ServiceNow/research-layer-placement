"""Tests for lower_confidence_bound acquisition function."""

import numpy as np

from place_layers.surrogate.acquisition import (
    expected_improvement,
    lower_confidence_bound,
)


class TestLCB:
    """Tests for lower_confidence_bound."""

    def test_lcb_penalizes_uncertainty(self) -> None:
        """Higher sigma → lower LCB score."""
        mu = np.array([1.0, 1.0])
        sigma = np.array([0.1, 2.0])
        lcb = lower_confidence_bound(mu, sigma, kappa=2.0)
        assert lcb[0] > lcb[1], "Higher uncertainty should give lower LCB"

    def test_lcb_kappa_zero_is_mean(self) -> None:
        """kappa=0 should return mu."""
        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.array([0.5, 1.0, 2.0])
        lcb = lower_confidence_bound(mu, sigma, kappa=0.0)
        np.testing.assert_array_equal(lcb, mu)

    def test_lcb_kappa_increases_penalty(self) -> None:
        """Higher kappa should give lower scores."""
        mu = np.array([1.0])
        sigma = np.array([1.0])
        lcb_low = lower_confidence_bound(mu, sigma, kappa=1.0)
        lcb_high = lower_confidence_bound(mu, sigma, kappa=3.0)
        assert lcb_low[0] > lcb_high[0]

    def test_lcb_shape(self) -> None:
        """Output should have same shape as input."""
        mu = np.array([1.0, 2.0, 3.0, 4.0])
        sigma = np.array([0.1, 0.2, 0.3, 0.4])
        lcb = lower_confidence_bound(mu, sigma, kappa=2.0)
        assert lcb.shape == (4,)


class TestEIUnchanged:
    """Verify EI function still works correctly."""

    def test_ei_basic(self) -> None:
        """EI should be non-negative."""
        mu = np.array([0.5, 1.5])
        sigma = np.array([0.5, 0.5])
        f_best = 1.0
        ei = expected_improvement(mu, sigma, f_best, xi=0.0)
        assert np.all(ei >= -1e-12)

    def test_ei_zero_sigma(self) -> None:
        """EI with sigma=0 should return max(mu - f_best, 0)."""
        mu = np.array([1.0, 2.0, 0.5])
        sigma = np.array([0.0, 0.0, 0.0])
        f_best = 1.0
        ei = expected_improvement(mu, sigma, f_best, xi=0.0)
        np.testing.assert_array_almost_equal(ei, [0.0, 1.0, 0.0])
