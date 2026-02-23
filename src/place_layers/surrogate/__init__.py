"""Surrogate model package for placement optimization."""

from place_layers.surrogate.acquisition import expected_improvement
from place_layers.surrogate.base_surrogate import BaseSurrogate, FrequentistCVDiagnostics, SurrogateDiagnostics
from place_layers.surrogate.bayesian_ridge import BayesianRidgeSurrogate
from place_layers.surrogate.config import (
    BaseSurrogateConfig,
    BayesianRidgeSurrogateConfig,
    RidgeSurrogateConfig,
    SurrogateConfig,
)
from place_layers.surrogate.dataset import ExplorationDataset
from place_layers.surrogate.evaluation_log import (
    EvaluationLogMeta,
    EvaluationRecord,
    PooledEvaluationData,
    append_record,
    discover_evaluation_files,
    load_pooled_data,
    load_records,
    resolve_supplemental_dirs,
)
from place_layers.surrogate.encoding import FeatureExpansionConfig
from place_layers.surrogate.forest_surrogate import ForestSurrogate
from place_layers.surrogate.model_selection import (
    ExpansionSelectionResult,
    StepResult,
    ridge_cv_rho,
    select_expansion_bayes_evidence,
    select_expansion_ridge_cv,
)
from place_layers.surrogate.ridge_surrogate import RidgeSurrogate
