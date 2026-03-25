"""Pydantic models for placement landscape data.

Follows the TraceDataset pattern: model_dump_json / model_validate_json
with .save() / .load() class methods.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class BandResult(BaseModel):
    """DP-optimal result for one cost band."""

    model_config = {"frozen": True}

    band_index: int
    cost_low: float
    cost_high: float
    cost: float
    score: float
    assignments: tuple[str, ...]


class PlacementResult(BaseModel):
    """A single placement with its cost and score."""

    model_config = {"frozen": True}

    cost: float
    score: float
    assignments: tuple[str, ...]


class BandValidationResult(BaseModel):
    """Per-band validated search results."""

    model_config = {"frozen": True}

    band_index: int
    cost_low: float
    cost_high: float
    validated: PlacementResult
    total_evaluations: int


class ValidationSummary(BaseModel):
    """Aggregate statistics across all validated bands."""

    model_config = {"frozen": True}

    total_bands: int
    total_evaluations: int


class ValidationResults(BaseModel):
    """Full output of validate_landscape command. Serializable to JSON."""

    model_config = {"frozen": True}

    objective_name: str = ""
    num_layers: int
    baseline_cost: float
    num_cost_bins: int
    cost_bin_edges: list[float]
    bands: list[BandValidationResult]
    summary: ValidationSummary

    def save(self, path: Path) -> None:
        """Save to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> ValidationResults:
        """Load from JSON file."""
        return cls.model_validate_json(path.read_text())


class LandscapeData(BaseModel):
    """Per-band DP results from placement_landscape. Serializable to JSON."""

    model_config = {"frozen": True}

    objective_name: str = ""
    num_layers: int
    baseline_cost: float
    num_cost_bins: int
    cost_bin_edges: list[float]
    bands: list[BandResult]

    def save(self, path: Path) -> None:
        """Save to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> LandscapeData:
        """Load from JSON file."""
        return cls.model_validate_json(path.read_text())
