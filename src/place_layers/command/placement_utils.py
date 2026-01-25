"""Utilities for placement resolution in commands."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from place_layers.core.types import LayerPlacement

if TYPE_CHECKING:
    from place_layers.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class PlacementResolution:
    """Result of resolving a placement specification."""

    def __init__(
        self,
        placement: LayerPlacement | None,
        placement_names: list[str] | None,
        placement_indices: list[int] | None,
    ) -> None:
        self.placement = placement
        self.placement_names = placement_names
        self.placement_indices = placement_indices


def resolve_placement(
    spec: list[str] | None,
    evaluator: "BaseEvaluator",
) -> PlacementResolution:
    """Resolve a placement specification to a LayerPlacement.

    Handles:
    - None -> no placement (for non-supernet models)
    - Cyclic extension if spec is shorter than num_layers
    - Validation of mixer names
    - Conversion from names to indices

    Args:
        spec: List of mixer names per layer, or None.
        evaluator: Evaluator providing model structure.

    Returns:
        PlacementResolution with placement object and metadata.

    Raises:
        ValueError: If placement is invalid (too long, unknown mixer names, etc.)
    """
    if spec is None:
        logger.info("No placement specified, using default configuration")
        return PlacementResolution(
            placement=None,
            placement_names=None,
            placement_indices=None,
        )

    # Get model structure
    num_layers = evaluator.num_layers
    mixer_names = evaluator.mixer_names

    logger.info(f"Model: {num_layers} layers, mixers: {list(mixer_names)}")

    # Check if model supports placement (multiple mixer types)
    if len(set(mixer_names)) == 1:
        raise ValueError(
            f"Placement specified but model only has one mixer type "
            f"({mixer_names[0]}). Use placement=null for non-supernet models."
        )

    # Build name-to-index mapping
    name_to_idx = {name: idx for idx, name in enumerate(mixer_names)}

    # Extend cyclically if shorter than num_layers (periodic boundary conditions)
    if len(spec) < num_layers:
        extended = [spec[i % len(spec)] for i in range(num_layers)]
        logger.info(f"Extended {len(spec)}-layer pattern to {num_layers} layers")
    elif len(spec) > num_layers:
        raise ValueError(
            f"Placement has {len(spec)} layers but model only has {num_layers}"
        )
    else:
        extended = list(spec)

    # Convert mixer names to indices
    try:
        indices = tuple(name_to_idx[name] for name in extended)
    except KeyError as e:
        raise ValueError(
            f"Unknown mixer name: {e}. Valid names: {list(name_to_idx.keys())}"
        ) from e

    logger.info(f"Placement: {extended[:6]}{'...' if len(extended) > 6 else ''}")

    return PlacementResolution(
        placement=LayerPlacement(assignments=indices),
        placement_names=extended,
        placement_indices=list(indices),
    )
