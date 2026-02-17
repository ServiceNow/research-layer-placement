"""Configuration for multiple evaluators with aggregation strategies.

This module provides a container for managing multiple evaluators simultaneously,
following the same pattern as ObserversConfig for search observers.

The key design is:
- Single `evaluator/` group with all individual evaluator configs (existing)
- New `evaluators/` group for presets composing multiple evaluators
- Container class (`EvaluatorsConfig`) with dict of named evaluators, weights, and aggregation
- Context-managed smart constructor

Use Cases
---------
1. Multi-metric evaluation with weighted average:
   - gsm8k (weight=0.6): generative evaluator on GSM8K
   - mmlu (weight=0.4): likelihood evaluator on MMLU
   - Aggregation: weighted_average
   - Combined score = 0.6 * gsm8k_score + 0.4 * mmlu_score

2. Multi-dataset optimization:
   - fda (weight=0.5): generative with extraction
   - gsm8k (weight=0.3): generative with reasoning
   - mmlu (weight=0.2): likelihood
   - Optimize for weighted combination of all three

3. Optimization with auxiliary metrics (no aggregation):
   - Use first evaluator for optimization
   - Others just for logging/tracking

YAML structure:
    evaluators:
      configs:
        gsm8k:
          evaluator_type: generative
          backend: ${backend}
          tasks: ["gsm8k"]
        mmlu:
          evaluator_type: likelihood
          backend: ${backend}
          tasks: ["mmlu"]
      weights:
        gsm8k: 0.6
        mmlu: 0.4
      aggregation: weighted_average  # or "first_only"

CLI usage:
    # Use single evaluator (backwards compatible)
    python -m place_layers +command=evaluate +evaluator=generative

    # Use multi-evaluator preset
    python -m place_layers +command=evaluate +evaluators=multi_gsm8k_mmlu

Code usage:
    async with config.evaluators.create_evaluators(output_dir) as evals:
        # Evaluate with all evaluators and aggregate
        score = await evals.evaluate(placement)

        # Or access individual evaluators
        for name, evaluator in evals.evaluators.items():
            score = await evaluator.evaluate(placement)
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, AsyncGenerator, Self

from pydantic import BaseModel, Field, model_validator

# Import at runtime for Pydantic (no circular import - this module isn't imported by evaluator/__init__.py)
from place_layers.evaluator import EvaluatorConfig

if TYPE_CHECKING:
    from place_layers.evaluator.base_evaluator import BaseEvaluator


class AggregationStrategy(str, Enum):
    """Strategy for aggregating scores from multiple evaluators."""

    WEIGHTED_AVERAGE = "weighted_average"
    """Weighted average: score = sum(weight_i * score_i)"""

    FIRST_ONLY = "first_only"
    """Use only the first evaluator's score (for auxiliary metrics)"""

    MIN = "min"
    """Take minimum score across all evaluators"""

    MAX = "max"
    """Take maximum score across all evaluators"""


class EvaluatorInstances[ParametersT]:
    """Container for instantiated evaluators with aggregation logic.

    Implements BaseEvaluator protocol to allow transparent use as a single evaluator.
    Provides access to individual evaluators and aggregated evaluation.

    Attributes:
        evaluators: Dict of named evaluators
        weights: Dict of weights for each evaluator (for weighted_average)
        aggregation: Strategy for combining scores
        _stack: AsyncExitStack managing evaluator lifecycles
    """

    def __init__(
        self,
        evaluators: dict[str, BaseEvaluator[ParametersT]],
        weights: dict[str, float],
        aggregation: AggregationStrategy,
        stack: AsyncExitStack,
    ) -> None:
        self.evaluators = evaluators
        self.weights = weights
        self.aggregation = aggregation
        self._stack = stack

        # Properties are lazily accessed from first evaluator
        # (can't cache in __init__ because evaluators haven't entered context yet)

    @property
    def num_layers(self) -> int:
        """Number of layers in the model (from first evaluator)."""
        first_evaluator = next(iter(self.evaluators.values()))
        return first_evaluator.num_layers

    @property
    def mixer_names(self) -> frozenset[str]:
        """Available mixer types (from first evaluator)."""
        first_evaluator = next(iter(self.evaluators.values()))
        return first_evaluator.mixer_names

    @property
    def num_mixers(self) -> int:
        """Number of mixer types."""
        return len(self.mixer_names)

    async def evaluate(self, parameters: ParametersT) -> dict[str, float]:
        """Evaluate parameters using all evaluators in parallel.

        All evaluators run concurrently for maximum performance. The shared
        backend handles request throttling via max_concurrent_requests.

        Args:
            parameters: Parameters to evaluate

        Returns:
            Dict mapping evaluator name to score
        """
        # Create evaluation tasks for all evaluators
        eval_coros = [
            evaluator.evaluate(parameters)
            for evaluator in self.evaluators.values()
        ]

        # Run all evaluators concurrently
        score_list = await asyncio.gather(*eval_coros)

        # Map results back to evaluator names
        return dict(zip(self.evaluators.keys(), score_list))

    async def evaluate_batch(self, parameters_list: list[ParametersT]) -> list[dict[str, float]]:
        """Evaluate multiple parameter settings in batch.

        Args:
            parameters_list: List of parameters to evaluate

        Returns:
            List of per-evaluator score dicts
        """
        return [await self.evaluate(p) for p in parameters_list]

    def aggregate(self, scores: dict[str, float]) -> float:
        """Aggregate scores from multiple evaluators.

        Args:
            scores: Dict mapping evaluator name to score

        Returns:
            Aggregated score
        """
        if self.aggregation == AggregationStrategy.WEIGHTED_AVERAGE:
            total = sum(self.weights[name] * score for name, score in scores.items())
            return total

        elif self.aggregation == AggregationStrategy.FIRST_ONLY:
            first_name = next(iter(self.evaluators.keys()))
            return scores[first_name]

        elif self.aggregation == AggregationStrategy.MIN:
            return min(scores.values())

        elif self.aggregation == AggregationStrategy.MAX:
            return max(scores.values())

        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation}")

    async def __aenter__(self) -> Self:
        """Enter context (evaluators already entered via stack)."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context and cleanup all evaluator resources."""
        # ExitStack handles cleanup of all evaluators
        await self._stack.aclose()


class EvaluatorsConfig(BaseModel):
    """Container for multiple evaluator configurations with aggregation.

    Manages multiple evaluators with weights and aggregation strategy.
    All evaluators are context managers by contract.

    Usage:
        async with config.evaluators.create_evaluators(output_dir) as evals:
            # Aggregate across all evaluators
            score = await evals.evaluate(placement)

            # Or access individual evaluators
            for name, evaluator in evals.evaluators.items():
                individual_score = await evaluator.evaluate(placement)
    """

    model_config = {"frozen": True}

    configs: dict[str, EvaluatorConfig] = Field(
        description="Dict of named evaluator configurations"
    )

    weights: dict[str, float] = Field(
        default_factory=dict,
        description="Weights for each evaluator (used with weighted_average). "
        "If empty, uniform weights are used.",
    )

    aggregation: AggregationStrategy = Field(
        default=AggregationStrategy.WEIGHTED_AVERAGE,
        description="Strategy for aggregating scores from multiple evaluators",
    )

    @model_validator(mode="after")
    def validate_weights(self) -> Self:
        """Validate that weights match evaluators and sum to 1.0."""
        if not self.configs:
            raise ValueError("Must provide at least one evaluator")

        # If weights not specified, use uniform weights
        if not self.weights:
            n = len(self.configs)
            object.__setattr__(
                self, "weights", {name: 1.0 / n for name in self.configs}
            )

        # Check that weight keys match evaluator keys
        if set(self.weights.keys()) != set(self.configs.keys()):
            raise ValueError(
                f"Weight keys {set(self.weights.keys())} must match "
                f"evaluator keys {set(self.configs.keys())}"
            )

        # Check that weights sum to 1.0 (only for weighted_average)
        if self.aggregation == AggregationStrategy.WEIGHTED_AVERAGE:
            total = sum(self.weights.values())
            if not (0.99 <= total <= 1.01):  # Allow small float errors
                raise ValueError(
                    f"Weights must sum to 1.0 for weighted_average, got {total:.4f}"
                )

        # Check that all weights are non-negative
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("All weights must be non-negative")

        return self

    @asynccontextmanager
    async def create_evaluators(
        self, output_dir: Path
    ) -> AsyncGenerator[EvaluatorInstances, None]:
        """Context manager: creates all evaluators and manages their lifecycle.

        Each evaluator gets its own output subdirectory:
        - {output_dir}/evals/{name}/

        Backend sharing: One backend per unique backend config. Evaluators
        with identical backend configs share a single backend instance.
        Non-backend evaluators (e.g., MockSupernetEvaluator) are created
        without a backend.

        Args:
            output_dir: Base output directory for all evaluators

        Yields:
            EvaluatorInstances containing all evaluators with aggregation logic

        Example:
            async with config.evaluators.create_evaluators(output_dir) as evals:
                # Evaluate and aggregate
                score = await evals.evaluate(placement)

                # Also access individual scores
                for name, evaluator in evals.evaluators.items():
                    individual_score = await evaluator.evaluate(placement)
                    logger.info(f"{name}: {individual_score:.4f}")
        """
        import json
        import logging

        from place_layers.backend.base import AsyncBackend, SyncBackend
        from place_layers.evaluator.backend_evaluator import BackendEvaluatorConfig

        logger = logging.getLogger(__name__)

        async with AsyncExitStack() as stack:
            # Create output directories
            evals_dir = output_dir / "evals"
            evals_dir.mkdir(parents=True, exist_ok=True)

            # Create one backend per unique backend config
            backend_cache: dict[str, AsyncBackend | SyncBackend] = {}
            for name, config in self.configs.items():
                if isinstance(config, BackendEvaluatorConfig):
                    key = json.dumps(
                        config.backend.model_dump(mode="json"), sort_keys=True
                    )
                    if key not in backend_cache:
                        backend = config.backend.create_backend()
                        if isinstance(backend, AsyncBackend):
                            await stack.enter_async_context(backend)
                        else:
                            stack.enter_context(backend)
                        backend_cache[key] = backend

            logger.info(
                "Created %d backend(s) for %d evaluator(s)",
                len(backend_cache),
                len(self.configs),
            )

            # Create all evaluators
            evaluators = {}
            for name, config in self.configs.items():
                eval_dir = evals_dir / name
                eval_dir.mkdir(parents=True, exist_ok=True)

                if isinstance(config, BackendEvaluatorConfig):
                    key = json.dumps(
                        config.backend.model_dump(mode="json"), sort_keys=True
                    )
                    evaluator = config.create_evaluator(
                        output_dir=eval_dir, backend=backend_cache[key]
                    )
                else:
                    evaluator = config.create_evaluator(output_dir=eval_dir)

                evaluators[name] = await stack.enter_async_context(evaluator)

            # Cross-evaluator validation: all backend evaluators must agree
            # on model structure (num_layers, mixer_names)
            backend_evaluators = [
                e for e in evaluators.values() if hasattr(e, "num_layers")
            ]
            if len(backend_evaluators) > 1:
                first = backend_evaluators[0]
                for e in backend_evaluators[1:]:
                    if e.num_layers != first.num_layers:
                        raise ValueError(
                            f"Evaluator num_layers mismatch: "
                            f"{first.num_layers} vs {e.num_layers}"
                        )
                    if e.mixer_names != first.mixer_names:
                        raise ValueError(
                            f"Evaluator mixer_names mismatch: "
                            f"{first.mixer_names} vs {e.mixer_names}"
                        )

            # Yield instances with aggregation logic
            instances = EvaluatorInstances(
                evaluators=evaluators,
                weights=self.weights,
                aggregation=self.aggregation,
                stack=stack,
            )
            yield instances
