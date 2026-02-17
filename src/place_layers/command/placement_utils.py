"""Utilities for placement resolution in commands."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from place_layers.core.types import LayerPlacement

if TYPE_CHECKING:
    from place_layers.evaluator.evaluators import Evaluators

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
    evaluators: "Evaluators",
) -> PlacementResolution:
    """Resolve a placement specification to a LayerPlacement.

    Handles:
    - None -> no placement (for non-supernet models)
    - Cyclic extension if spec is shorter than num_layers
    - Validation of mixer names
    - Conversion from names to indices

    Args:
        spec: List of mixer names per layer, or None.
        evaluators: Evaluators providing model structure.

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
    num_layers = evaluators.num_layers
    mixer_names = evaluators.mixer_names

    logger.info(f"Model: {num_layers} layers, mixers: {list(mixer_names)}")

    # Check if model supports placement (multiple mixer types)
    if len(mixer_names) == 1:
        only_mixer = next(iter(mixer_names))
        raise ValueError(
            f"Placement specified but model only has one mixer type "
            f"({only_mixer}). Use placement=null for non-supernet models."
        )

    # Build valid mixer names set for validation
    valid_names = set(mixer_names)

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

    # Validate mixer names
    for name in extended:
        if name not in valid_names:
            raise ValueError(
                f"Unknown mixer name: {name}. Valid names: {list(valid_names)}"
            )

    logger.info(f"Placement: {extended[:6]}{'...' if len(extended) > 6 else ''}")

    # Build name-to-index mapping for backwards compatibility indices
    name_to_idx = {name: idx for idx, name in enumerate(mixer_names)}
    indices = [name_to_idx[name] for name in extended]

    return PlacementResolution(
        placement=LayerPlacement(assignments=tuple(extended)),
        placement_names=extended,
        placement_indices=indices,
    )
