"""Shared configuration for landscape analysis commands.

Bundled as a Hydra config group so validate_landscape and
placement_landscape use structurally identical binning and surrogate
parameters.  Impossible to have different num_cost_bins or seed between
the two steps when both embed ``landscape: ${landscape}``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from place_layers.cost_model import CostModelConfig


class LandscapeConfig(BaseModel):
    """Shared configuration for landscape analysis commands.

    All defaults live in YAML (e.g. conf/landscape/apriel_15b.yaml),
    not here.
    """

    model_config = {"frozen": True}

    cost_model: CostModelConfig = Field(
        description="Cost model for allocation enumeration and cost computation.",
    )
    exploration_dataset_path: str = Field(
        description="Path to exploration_results.json from explore_placements.",
    )
    seed: int = Field(
        description="Random seed for surrogate fitting and model selection.",
    )
    num_cost_bins: int = Field(
        description="Number of cost bins for band computation.",
        gt=0,
    )
