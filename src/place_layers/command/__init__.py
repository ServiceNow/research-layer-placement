"""Command pattern implementation for place_layers."""

from typing import Annotated

from pydantic import Field, TypeAdapter

from place_layers.command.calibrate_sa import CalibrateSACommandConfig
from place_layers.command.convergence_analysis import ConvergenceAnalysisCommandConfig
from place_layers.command.evaluate import EvaluateCommandConfig
from place_layers.command.explore_placements import ExplorePlacementsCommandConfig
from place_layers.command.hyperparameter_analysis import (
    HyperparameterAnalysisCommandConfig,
)
from place_layers.command.landscape_analysis import LandscapeAnalysisCommandConfig
from place_layers.command.noise_analysis import NoiseAnalysisCommandConfig
from place_layers.command.optimize import OptimizeCommandConfig
from place_layers.command.placement_landscape import PlacementLandscapeCommandConfig
from place_layers.command.precompute_traces import PrecomputeTracesCommandConfig
from place_layers.command.sensitivity_analysis import SensitivityAnalysisCommandConfig
from place_layers.command.serve import ServeCommandConfig
from place_layers.command.validate_landscape import ValidateLandscapeCommandConfig


# Discriminated union of all commands
CommandConfig = Annotated[
    OptimizeCommandConfig
    | LandscapeAnalysisCommandConfig
    | ConvergenceAnalysisCommandConfig
    | SensitivityAnalysisCommandConfig
    | HyperparameterAnalysisCommandConfig
    | CalibrateSACommandConfig
    | PrecomputeTracesCommandConfig
    | EvaluateCommandConfig
    | ServeCommandConfig
    | NoiseAnalysisCommandConfig
    | ExplorePlacementsCommandConfig
    | PlacementLandscapeCommandConfig
    | ValidateLandscapeCommandConfig,
    Field(discriminator="command"),
]

# Type adapter for validation
command_adapter: TypeAdapter[CommandConfig] = TypeAdapter(CommandConfig)
