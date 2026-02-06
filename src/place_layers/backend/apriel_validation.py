"""Validation and structure extraction for Apriel2 supernet models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class Apriel2ValidationError(ValueError):
    """Model is not a valid Apriel2 supernet."""


class Apriel2Structure(BaseModel):
    """Extracted structure from Apriel2 supernet config."""

    model_config = {"frozen": True}

    num_layers: int
    mixer_names: tuple[str, ...]
    stochastic_block_name: str
    pattern_type: str


def validate_apriel2_model(model: "PreTrainedModel") -> Apriel2Structure:
    """Validate Apriel2 supernet and extract structure.

    Checks config class name (handles trust_remote_code loading a separate class),
    then walks the decoder dict.
    """
    config_class_name = type(model.config).__name__
    if config_class_name not in ("Apriel2TextConfig", "Apriel2Config"):
        raise Apriel2ValidationError(
            f"Expected Apriel2TextConfig or Apriel2Config, got {config_class_name}"
        )

    decoder = model.config.decoder

    num_blocks = decoder["num_blocks"]

    match decoder["type"]:
        case "fixed":
            return _validate_fixed(decoder, num_blocks)
        case "pattern":
            return _validate_pattern(decoder, num_blocks)
        case other:
            raise Apriel2ValidationError(f"Unknown decoder type: {other}")


def _validate_fixed(decoder: dict, num_blocks: int) -> Apriel2Structure:
    """Fixed: single block repeated for all layers."""
    mixer = decoder["block"]["mixer"]

    if mixer["type"] != "stochastic":
        raise Apriel2ValidationError(
            f"Block mixer type is '{mixer['type']}', expected 'stochastic'"
        )

    return Apriel2Structure(
        num_layers=num_blocks,
        mixer_names=tuple(sorted(mixer["mixers"].keys())),
        stochastic_block_name="block",
        pattern_type="fixed",
    )


def _validate_pattern(decoder: dict, num_blocks: int) -> Apriel2Structure:
    """Pattern: find the one block type with stochastic mixer."""
    pattern = decoder["pattern"]
    blocks = decoder["blocks"]

    # Find blocks with stochastic mixer
    stochastic = [
        (name, cfg["mixer"])
        for name, cfg in blocks.items()
        if cfg["mixer"]["type"] == "stochastic"
    ]

    if len(stochastic) == 0:
        raise Apriel2ValidationError("No block has stochastic mixer")
    if len(stochastic) > 1:
        raise Apriel2ValidationError(
            f"Multiple blocks have stochastic mixer: {[n for n, _ in stochastic]}"
        )

    block_name, mixer = stochastic[0]

    # Count occurrences in pattern
    count_per_cycle = pattern.count(block_name)
    full_cycles = num_blocks // len(pattern)
    remainder_count = pattern[: num_blocks % len(pattern)].count(block_name)

    return Apriel2Structure(
        num_layers=full_cycles * count_per_cycle + remainder_count,
        mixer_names=tuple(sorted(mixer["mixers"].keys())),
        stochastic_block_name=block_name,
        pattern_type="pattern",
    )
