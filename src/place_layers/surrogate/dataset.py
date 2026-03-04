"""Serializable dataset of evaluated placements for surrogate training."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from place_layers.surrogate.config import SurrogateConfig
from place_layers.surrogate.encoding import FeatureExpansionConfig
from place_layers.surrogate.model_selection import ExpansionSelectionResult


class ExplorationDataset(BaseModel):
    """Serializable dataset of evaluated placements for a single objective.

    Each dataset stores scalar scores for one objective (identified by
    ``objective_name``).  When the optimizer has M objectives, the explore
    command produces M independent datasets.
    """

    model_config = {"frozen": True, "extra": "ignore"}

    # Data
    assignments: list[tuple[str, ...]]
    scores: list[float]

    @property
    def allocations(self) -> list[dict[str, int]]:
        """Compute allocations from assignments (never stored)."""
        return [dict(Counter(a)) for a in self.assignments]

    # Which objective this dataset is for
    objective_name: str

    # Metadata
    num_layers: int
    mixer_names: list[str]
    num_samples: int
    termination_reason: Literal["target_reached", "plateau", "max_samples"]

    # Feature expansion config — required (no backward compat)
    feature_config: FeatureExpansionConfig

    # Surrogate provenance (None for backward compat with pre-existing JSON)
    surrogate_config: SurrogateConfig | None = None
    expansion_result: ExpansionSelectionResult | None = None

    # Learning curve
    rho_history: list[float]
    samples_at_check: list[int]
    final_rho: float

    @classmethod
    def load(cls, path: Path) -> "ExplorationDataset":
        """Load dataset from JSON file."""
        return cls.model_validate_json(path.read_text())
