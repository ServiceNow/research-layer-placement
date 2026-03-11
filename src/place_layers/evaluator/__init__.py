"""Evaluator implementations."""

from typing import Annotated

from pydantic import Field

from place_layers.evaluator.estimate_acceptance_rate import AcceptanceRateEvaluatorConfig
from place_layers.evaluator.generative import GenerativeEvaluatorConfig
from place_layers.evaluator.importance_weighted import ImportanceWeightedEvaluatorConfig
from place_layers.evaluator.likelihood import LikelihoodEvaluatorConfig
from place_layers.evaluator.mock_supernet_evaluator import MockSupernetEvaluatorConfig
from place_layers.evaluator.speculative import SpeculativeEvaluatorConfig
from place_layers.evaluator.throughput import ThroughputEvaluatorConfig
from place_layers.evaluator.trace_likelihood import TraceLikelihoodEvaluatorConfig
from place_layers.evaluator.trace_precompute import TracePrecomputeEvaluatorConfig


# Discriminated union of all evaluators
EvaluatorConfig = Annotated[
    MockSupernetEvaluatorConfig
    | ImportanceWeightedEvaluatorConfig
    | GenerativeEvaluatorConfig
    | LikelihoodEvaluatorConfig
    | TracePrecomputeEvaluatorConfig
    | SpeculativeEvaluatorConfig
    | TraceLikelihoodEvaluatorConfig
    | TracePrecomputeEvaluatorConfig
    | AcceptanceRateEvaluatorConfig
    | ThroughputEvaluatorConfig,
    Field(discriminator="evaluator_type"),
]
