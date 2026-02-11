"""Ordinal + allocation-count encoding for layer placements.

Each placement is encoded as a float32 vector of length (num_layers + num_types):
  - First num_layers features: ordinal encoding of each layer's mixer type
  - Last num_types features: count of layers assigned to each mixer type

The ordinal mapping is built from sorted mixer names for determinism.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


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


def encode_config(
    assignments: tuple[str, ...] | list[str],
    type_to_int: dict[str, int],
) -> np.ndarray:
    """Encode a single placement as a feature vector.

    Args:
        assignments: Mixer name per layer.
        type_to_int: Ordinal mapping from mixer name to int.

    Returns:
        Float32 vector of shape (num_layers + num_types,).
    """
    num_layers = len(assignments)
    num_types = len(type_to_int)

    features = np.empty(num_layers + num_types, dtype=np.float32)

    # Ordinal encoding for each layer position
    for i, mixer in enumerate(assignments):
        features[i] = type_to_int[mixer]

    # Allocation counts
    counts = Counter(assignments)
    for mixer, idx in type_to_int.items():
        features[num_layers + idx] = counts.get(mixer, 0)

    return features


def encode_batch(
    configs: list[tuple[str, ...] | list[str]],
    type_to_int: dict[str, int],
) -> np.ndarray:
    """Encode a batch of placements.

    Args:
        configs: List of placements (each is a sequence of mixer names).
        type_to_int: Ordinal mapping.

    Returns:
        Float32 matrix of shape (n, num_layers + num_types).
    """
    if not configs:
        num_types = len(type_to_int)
        return np.empty((0, num_types), dtype=np.float32)

    rows = [encode_config(cfg, type_to_int) for cfg in configs]
    return np.stack(rows)


def decode_config(
    features: np.ndarray,
    int_to_type: dict[int, str],
    num_layers: int,
) -> tuple[str, ...]:
    """Decode a feature vector back to a placement.

    Only the first num_layers features (ordinal positions) are used;
    count features are ignored (they are redundant).

    Args:
        features: Feature vector of shape (num_layers + num_types,).
        int_to_type: Reverse ordinal mapping.
        num_layers: Number of layers.

    Returns:
        Tuple of mixer names.
    """
    assignments = []
    for i in range(num_layers):
        idx = int(round(features[i]))
        assignments.append(int_to_type[idx])
    return tuple(assignments)


def feature_names(
    num_layers: int,
    int_to_type: dict[int, str],
) -> list[str]:
    """Human-readable feature names for diagnostics.

    Args:
        num_layers: Number of layers.
        int_to_type: Reverse ordinal mapping.

    Returns:
        List of feature names, one per feature dimension.
    """
    names = [f"layer_{i}" for i in range(num_layers)]
    for idx in sorted(int_to_type.keys()):
        names.append(f"count_{int_to_type[idx]}")
    return names
