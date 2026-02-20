"""Placement landscape analysis infrastructure.

Provides Pydantic models, analysis functions, and shared configuration
used by the landscape pipeline commands (placement_landscape, validate_landscape).
"""

from place_layers.landscape.analysis import (
    compute_band_edges,
    compute_dp_per_band,
    load_and_fit_surrogate,
)
from place_layers.landscape.config import LandscapeConfig, objective_result_path
from place_layers.landscape.models import (
    BandResult,
    BandValidationResult,
    LandscapeData,
    PlacementResult,
    ValidationResults,
    ValidationSummary,
)
