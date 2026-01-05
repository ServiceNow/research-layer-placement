"""Evaluator implementations."""

from typing import Annotated

from pydantic import Field

from place_layers.evaluator.generative import GenerativeEvaluatorConfig
from place_layers.evaluator.importance_weighted import (
    ImportanceWeightedEvaluatorConfig,
)
from place_layers.evaluator.mock_supernet_evaluator import MockSupernetEvaluatorConfig


# Discriminated union of all evaluators
EvaluatorConfig = Annotated[
    MockSupernetEvaluatorConfig
    | ImportanceWeightedEvaluatorConfig
    | GenerativeEvaluatorConfig,
    Field(discriminator="evaluator_type"),
]
