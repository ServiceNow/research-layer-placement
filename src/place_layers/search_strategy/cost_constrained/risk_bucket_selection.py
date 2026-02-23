"""Risk-bucket candidate selection for Bayesian surrogate search.

Selects candidates from a scored pool using configurable risk buckets.
Each bucket scores candidates as ``mu + sign * beta * sigma`` and selects
top-scoring ones, with optional mean-floor filtering.

This is a pure function of (mu, sigma, n_select, risk_buckets) — fully
independent of how the pool was generated.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field


class RiskBucketConfig(BaseModel, frozen=True):
    """Configuration for a single risk bucket used in candidate selection.

    Each bucket scores candidates as ``mu + sign * beta * sigma`` and selects
    top-scoring ones.  The ``mean_floor_quantile`` optionally filters out
    candidates whose predicted mean is below the given quantile of the pool.
    """

    name: str = Field(description="Human-readable bucket name (e.g. 'safe', 'upside').")
    sign: Literal[-1, 1] = Field(
        description="Score sign: -1 for conservative (penalise uncertainty), +1 for optimistic.",
    )
    betas: list[float] = Field(
        description="Non-negative beta multipliers for sigma. Non-empty.",
    )
    quota: float = Field(
        description="Fraction of the per-round evaluation budget allocated to this bucket.",
        ge=0.0,
    )
    mean_floor_quantile: float = Field(
        description="Quantile of the pool mean predictions used as eligibility floor. "
                    "0.0 means no floor (all candidates eligible).",
        ge=0.0,
        le=1.0,
    )


def select_by_risk_buckets(
    mu: np.ndarray,
    sigma: np.ndarray,
    n_select: int,
    risk_buckets: list[RiskBucketConfig],
) -> tuple[list[int], list[str]]:
    """Select candidate indices using risk-bucket apportionment.

    Args:
        mu: Predicted means, shape (n,).
        sigma: Predicted standard deviations, shape (n,).
        n_select: Number of candidates to select.
        risk_buckets: Risk bucket configurations.

    Returns:
        (selected_indices, log_lines)
    """
    selected: list[int] = []
    selected_set: set[int] = set()

    # Largest-remainder apportionment across buckets
    quotas = [b.quota for b in risk_buckets]
    raw = [q * n_select for q in quotas]
    base = [int(np.floor(r)) for r in raw]
    target_total = min(n_select, int(np.floor(sum(raw) + 1e-9)))
    left = max(0, target_total - sum(base))
    remainders = [(raw[i] - base[i], i) for i in range(len(base))]
    remainders.sort(reverse=True)
    for _, i in remainders[:left]:
        base[i] += 1

    log_lines: list[str] = []

    for bucket_idx, bucket in enumerate(risk_buckets):
        k_bucket = base[bucket_idx]
        if k_bucket == 0:
            continue

        # Split k_bucket across betas with exact accounting
        kb = k_bucket // len(bucket.betas)
        kr = k_bucket % len(bucket.betas)

        # mean_floor_quantile: eligible candidates have mu >= quantile(mu, q)
        floor_val = float(np.quantile(mu, bucket.mean_floor_quantile))

        remaining_k_bucket = k_bucket

        for j, beta in enumerate(bucket.betas):
            k = kb + (1 if j < kr else 0)
            if k == 0:
                continue
            eligible = [i for i in range(len(mu))
                        if i not in selected_set and mu[i] >= floor_val]
            if not eligible:
                continue
            eligible_arr = np.array(eligible)
            scores = mu[eligible_arr] + bucket.sign * beta * sigma[eligible_arr]
            order = np.argsort(scores, kind="mergesort")[::-1][:k]
            added = 0
            for idx in order:
                if remaining_k_bucket <= 0 or len(selected) >= n_select:
                    break
                i = int(eligible_arr[idx])
                selected.append(i)
                selected_set.add(i)
                remaining_k_bucket -= 1
                added += 1
            log_lines.append(
                f"  {bucket.name}(sign={bucket.sign}, \u03b2={beta}): "
                f"selected {added}, "
                f"mean(\u03bc)={float(np.mean(mu[eligible_arr])):.4f}, "
                f"mean(\u03c3)={float(np.mean(sigma[eligible_arr])):.4f}"
            )

        # Within-bucket backfill: if some beta sub-buckets had no eligible
        # candidates, fill remaining_k_bucket using beta=0 (pure mu)
        if remaining_k_bucket > 0:
            eligible = [i for i in range(len(mu))
                        if i not in selected_set and mu[i] >= floor_val]
            if eligible:
                eligible_arr = np.array(eligible)
                order = np.argsort(mu[eligible_arr], kind="mergesort")[::-1]
                backfilled = 0
                for idx in order:
                    if remaining_k_bucket <= 0 or len(selected) >= n_select:
                        break
                    i = int(eligible_arr[idx])
                    selected.append(i)
                    selected_set.add(i)
                    remaining_k_bucket -= 1
                    backfilled += 1
                if backfilled > 0:
                    log_lines.append(
                        f"  {bucket.name}(backfill \u03b2=0): {backfilled}"
                    )

    # Global backfill by mu desc
    if len(selected) < n_select:
        remaining = [i for i in range(len(mu)) if i not in selected_set]
        remaining_arr = np.array(remaining) if remaining else np.array([], dtype=int)
        if len(remaining_arr) > 0:
            order = np.argsort(mu[remaining_arr], kind="mergesort")[::-1]
            global_backfilled = 0
            for idx in order:
                if len(selected) >= n_select:
                    break
                selected.append(int(remaining_arr[idx]))
                selected_set.add(int(remaining_arr[idx]))  # keep set consistent
                global_backfilled += 1
            if global_backfilled > 0:
                log_lines.append(
                    f"  global backfill (\u03bc desc): {global_backfilled}"
                )

    return selected, log_lines
