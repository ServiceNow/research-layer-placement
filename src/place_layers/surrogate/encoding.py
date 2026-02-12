"""One-hot + cluster-expansion encoding for layer placements.

Each placement is encoded as a float32 vector with configurable feature depth:

Level 0 (max_order=1): Unary one-hot + allocation counts
  - N*K one-hot features + K count features

Level 1+ (max_order>=2): + pairwise indicators for ranges 1..pairwise_max_range
  - For each range r: (N-r)*K^2 indicators

Level 2+ (max_order>=3): + n-body consecutive tuple indicators for orders 3..max_order
  - For each order n: (N-n+1)*K^n indicators

The type mapping is built from sorted mixer names for determinism.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel


class FeatureExpansionConfig(BaseModel):
    """Configuration for cluster expansion feature depth.

    Controls which n-body interaction terms are included in the feature vector.
    Higher orders capture more complex interactions but grow as O(N * K^n).
    """

    model_config = {"frozen": True}

    max_order: int = 1
    """Maximum n-body order. 1=unary only, 2=+pairwise, 3=+triplets, etc."""

    pairwise_max_range: int = 1
    """Max skip distance for order-2 terms (ignored if max_order < 2)."""

    allocation_counts: bool = True
    """Whether to append allocation count features."""


def build_type_mapping(
    mixer_names: list[str],
) -> tuple[dict[str, int], dict[int, str]]:
    """Build ordinal mappings from mixer names, sorted alphabetically.

    Args:
        mixer_names: List of mixer type names.

    Returns:
        (type_to_int, int_to_type) mapping dicts.
    """
    sorted_names = sorted(mixer_names)
    type_to_int = {name: i for i, name in enumerate(sorted_names)}
    int_to_type = {i: name for i, name in enumerate(sorted_names)}
    return type_to_int, int_to_type


def compute_feature_dim(
    num_layers: int,
    num_types: int,
    feature_config: FeatureExpansionConfig | None = None,
) -> int:
    """Compute total feature dimension without encoding.

    Args:
        num_layers: Number of layers (N).
        num_types: Number of mixer types (K).
        feature_config: Expansion config. None means default (unary + counts).

    Returns:
        Total number of features.
    """
    if feature_config is None:
        feature_config = FeatureExpansionConfig()

    N, K = num_layers, num_types

    # Unary one-hot
    dim = N * K

    # Pairwise indicators
    if feature_config.max_order >= 2:
        for r in range(1, feature_config.pairwise_max_range + 1):
            dim += max(0, N - r) * K * K

    # Higher-order consecutive tuple indicators (orders 3+)
    for n in range(3, feature_config.max_order + 1):
        dim += max(0, N - n + 1) * (K ** n)

    # Allocation counts
    if feature_config.allocation_counts:
        dim += K

    return dim


def encode_config(
    assignments: Sequence[str],
    type_to_int: dict[str, int],
    feature_config: FeatureExpansionConfig | None = None,
) -> np.ndarray:
    """Encode a single placement as a feature vector.

    Args:
        assignments: Mixer name per layer.
        type_to_int: Type mapping from mixer name to int.
        feature_config: Expansion config. None means default (unary + counts).

    Returns:
        Float32 feature vector.
    """
    if feature_config is None:
        feature_config = FeatureExpansionConfig()

    N = len(assignments)
    K = len(type_to_int)
    dim = compute_feature_dim(N, K, feature_config)
    features = np.zeros(dim, dtype=np.float32)

    # Convert to integer types once
    types = [type_to_int[a] for a in assignments]

    offset = 0

    # Unary one-hot encoding for each layer position
    for i in range(N):
        features[offset + i * K + types[i]] = 1.0
    offset += N * K

    # Pairwise indicators
    if feature_config.max_order >= 2:
        K2 = K * K
        for r in range(1, feature_config.pairwise_max_range + 1):
            for i in range(N - r):
                flat_idx = types[i] * K + types[i + r]
                features[offset + flat_idx] = 1.0
                offset += K2

    # Higher-order consecutive tuple indicators (orders 3+)
    for n in range(3, feature_config.max_order + 1):
        K_n = K ** n
        for i in range(N - n + 1):
            flat_idx = 0
            for j in range(n):
                flat_idx = flat_idx * K + types[i + j]
            features[offset + flat_idx] = 1.0
            offset += K_n

    # Allocation counts
    if feature_config.allocation_counts:
        counts = Counter(assignments)
        for mixer, idx in type_to_int.items():
            features[offset + idx] = counts.get(mixer, 0)

    return features


def encode_batch(
    configs: Sequence[Sequence[str]],
    type_to_int: dict[str, int],
    feature_config: FeatureExpansionConfig | None = None,
) -> np.ndarray:
    """Encode a batch of placements.

    Args:
        configs: List of placements (each is a sequence of mixer names).
        type_to_int: Type mapping.
        feature_config: Expansion config. None means default (unary + counts).

    Returns:
        Float32 matrix of shape (n, feature_dim).
    """
    if not configs:
        num_types = len(type_to_int)
        dim = compute_feature_dim(0, num_types, feature_config)
        return np.empty((0, dim), dtype=np.float32)

    rows = [encode_config(cfg, type_to_int, feature_config) for cfg in configs]
    return np.stack(rows)


def decode_config(
    features: np.ndarray,
    int_to_type: dict[int, str],
    num_layers: int,
) -> tuple[str, ...]:
    """Decode a feature vector back to a placement.

    Only the first num_layers * num_types features (one-hot blocks) are used;
    interaction and count features are ignored (they are redundant).

    Args:
        features: Feature vector.
        int_to_type: Reverse type mapping.
        num_layers: Number of layers.

    Returns:
        Tuple of mixer names.
    """
    num_types = len(int_to_type)
    assignments = []
    for i in range(num_layers):
        block = features[i * num_types : (i + 1) * num_types]
        idx = int(np.argmax(block))
        assignments.append(int_to_type[idx])
    return tuple(assignments)


def feature_names(
    num_layers: int,
    int_to_type: dict[int, str],
    feature_config: FeatureExpansionConfig | None = None,
) -> list[str]:
    """Human-readable feature names for diagnostics.

    Args:
        num_layers: Number of layers.
        int_to_type: Reverse ordinal mapping.
        feature_config: Expansion config. None means default (unary + counts).

    Returns:
        List of feature names, one per feature dimension.
    """
    if feature_config is None:
        feature_config = FeatureExpansionConfig()

    N = num_layers
    K = len(int_to_type)
    sorted_indices = sorted(int_to_type.keys())
    names: list[str] = []

    # Unary one-hot names
    for i in range(N):
        for idx in sorted_indices:
            names.append(f"layer_{i}_{int_to_type[idx]}")

    # Pairwise interaction names
    if feature_config.max_order >= 2:
        for r in range(1, feature_config.pairwise_max_range + 1):
            for i in range(N - r):
                for a in sorted_indices:
                    for b in sorted_indices:
                        names.append(
                            f"pair_{i}_{i + r}_{int_to_type[a]}_{int_to_type[b]}"
                        )

    # Higher-order consecutive tuple names (orders 3+)
    for n in range(3, feature_config.max_order + 1):
        for i in range(N - n + 1):
            # Generate all K^n type tuples in mixed-radix order
            for flat_idx in range(K ** n):
                type_tuple = []
                idx_remaining = flat_idx
                for _ in range(n):
                    idx_remaining, rem = divmod(idx_remaining, K)
                    type_tuple.append(rem)
                type_tuple.reverse()
                type_names = "_".join(int_to_type[t] for t in type_tuple)
                names.append(f"cluster{n}_{i}_{type_names}")

    # Allocation count names
    if feature_config.allocation_counts:
        for idx in sorted_indices:
            names.append(f"count_{int_to_type[idx]}")

    return names
