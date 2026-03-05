"""Shared configuration for landscape analysis commands.

Bundled as a Hydra config group so validate_landscape and
placement_landscape use structurally identical binning and surrogate
parameters.  Impossible to have different num_cost_bins or seed between
the two steps when both embed ``landscape: ${landscape}``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from place_layers.cost_model import CostModelConfig
from place_layers.surrogate.config import SurrogateConfig


def objective_result_path(directory: Path, prefix: str, objective_name: str) -> Path:
    """Deterministic per-objective result path.  Shared by all commands.

    Examples::

        objective_result_path(out, "exploration_results", "gsm8k")
        # -> out / "gsm8k" / "exploration_results.json"
    """
    return directory / objective_name / f"{prefix}.json"


class LandscapeConfig(BaseModel):
    """Shared configuration for landscape analysis commands.

    All defaults live in YAML (e.g. conf/landscape/apriel_15b.yaml),
    not here.
    """

    model_config = {"frozen": True}

    cost_model: CostModelConfig = Field(
        description="Cost model for allocation enumeration and cost computation.",
    )
    explore_output_dir: str = Field(
        description="Path to the output directory from explore_placements "
        "(contains {obj_name}/exploration_results.json files).",
    )
    seed: int = Field(
        description="Random seed for surrogate fitting and model selection.",
    )
    num_cost_bins: int = Field(
        description="Number of cost bins for band computation.",
        gt=0,
    )
    surrogate: SurrogateConfig = Field(
        description="Surrogate model configuration for DP solving and density prediction.",
    )
    supplemental_data_dirs: list[str] = Field(
        description="Directories containing additional evaluation data to merge "
        "into surrogate training. Each directory is searched per-objective "
        "using the standard naming convention ({obj}/evaluation_records.jsonl, etc.).",
    )
