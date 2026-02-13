"""Serializable dataset of evaluated placements for surrogate training."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from place_layers.surrogate.encoding import FeatureExpansionConfig


class ExplorationDataset(BaseModel):
    """Serializable dataset of evaluated placements."""

    model_config = {"frozen": True}

    # Data
    assignments: list[tuple[str, ...]]
    scores: list[float]
    allocations: list[dict[str, int]]

    # Metadata
    num_layers: int
    mixer_names: list[str]
    num_samples: int
    termination_reason: Literal["target_reached", "plateau", "max_samples"]

    # Feature expansion (None = Level 0 / unary only)
    feature_config: FeatureExpansionConfig | None = None

    # Learning curve
    rho_history: list[float]
    samples_at_check: list[int]
    final_rho: float

    @classmethod
    def load(cls, path: Path) -> "ExplorationDataset":
        """Load dataset from JSON file."""
        return cls.model_validate_json(path.read_text())
