"""Serializable dataset of evaluated placements for surrogate training."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

from place_layers.surrogate.encoding import FeatureExpansionConfig


class ExplorationDataset(BaseModel):
    """Serializable dataset of evaluated placements.

    Scores are always stored as per-evaluator dicts. For single-evaluator runs
    this is ``[{"main": 0.85}, ...]``. Multi-evaluator runs store the full
    breakdown: ``[{"mmlu": 0.85, "gsm8k": 0.72}, ...]``.

    Legacy datasets with ``scores: list[float]`` are auto-migrated on load
    by wrapping each scalar as ``{"main": score}``.
    """

    model_config = {"frozen": True}

    # Data
    assignments: list[tuple[str, ...]]
    scores: list[dict[str, float]]
    allocations: list[dict[str, int]]

    # Aggregation metadata (how to combine per-evaluator scores)
    evaluator_weights: dict[str, float] = {}
    aggregation: str = "weighted_average"

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

    @model_validator(mode="before")
    @classmethod
    def _migrate_scalar_scores(cls, data: dict) -> dict:
        """Auto-migrate legacy ``scores: list[float]`` to dict format."""
        if isinstance(data, dict) and "scores" in data:
            scores = data["scores"]
            if scores and isinstance(scores[0], (int, float)):
                data = dict(data)  # don't mutate caller's dict
                data["scores"] = [{"main": float(s)} for s in scores]
        return data

    def aggregated_scores(self) -> list[float]:
        """Compute aggregated scalar scores using stored weights.

        Falls back to uniform weights if ``evaluator_weights`` is empty
        (backward-compat for datasets created before the field existed).
        """
        weights = self.evaluator_weights
        agg = self.aggregation

        if agg == "first_only":
            first_key = next(iter(self.scores[0]))
            return [s[first_key] for s in self.scores]
        elif agg == "min":
            return [min(s.values()) for s in self.scores]
        elif agg == "max":
            return [max(s.values()) for s in self.scores]
        else:
            # weighted_average (default)
            if weights:
                return [
                    sum(weights[k] * s[k] for k in s) for s in self.scores
                ]
            # No weights stored → uniform average
            return [
                sum(s.values()) / len(s) for s in self.scores
            ]

    @classmethod
    def load(cls, path: Path) -> "ExplorationDataset":
        """Load dataset from JSON file."""
        return cls.model_validate_json(path.read_text())
