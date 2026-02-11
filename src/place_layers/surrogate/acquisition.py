"""Expected Improvement acquisition function for surrogate-assisted search."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Compute Expected Improvement for maximization.

    EI(x) = (mu - f_best - xi) * Phi(Z) + sigma * phi(Z)
    where Z = (mu - f_best - xi) / sigma

    Args:
        mu: Predicted means, shape (n,).
        sigma: Predicted standard deviations, shape (n,).
        f_best: Best observed value so far.
        xi: Exploration parameter (higher = more exploration).

    Returns:
        EI values, shape (n,). Non-negative.
    """
    ei = np.zeros_like(mu)

    # Only compute where sigma > 0 (non-degenerate predictions)
    mask = sigma > 1e-12
    if np.any(mask):
        improvement = mu[mask] - f_best - xi
        Z = improvement / sigma[mask]
        ei[mask] = improvement * norm.cdf(Z) + sigma[mask] * norm.pdf(Z)

    # Where sigma = 0: pure exploitation fallback
    no_var = ~mask
    if np.any(no_var):
        ei[no_var] = np.maximum(mu[no_var] - f_best, 0.0)

    return ei
