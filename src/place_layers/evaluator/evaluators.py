"""Primary evaluation interface: wraps 1+ evaluators with aggregation.

``Evaluators`` is the **only** evaluation interface that problems and search
strategies interact with.  Even a single evaluator is wrapped in this
container.  Individual ``BaseEvaluator`` subclasses stay clean — evaluate one
thing, return a float, know nothing about aggregation.

Configuration
-------------
.. code-block:: yaml

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
      aggregation: weighted_average

Code usage::

    async with config.evaluators.create_evaluators(output_dir, backend=backend) as evals:
        scores = await evals.evaluate(placement)       # dict[str, float]
        scalar = evals.aggregate(scores)               # float
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, AsyncGenerator, Self

from pydantic import BaseModel, Field, model_validator

from place_layers.evaluator import EvaluatorConfig

if TYPE_CHECKING:
    from place_layers.backend.base import AsyncBackend, SyncBackend
    from place_layers.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aggregation strategy
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Evaluators (runtime container)
# ---------------------------------------------------------------------------


class Evaluators[ParametersT]:
    """Primary evaluation interface.  Wraps 1+ evaluators with aggregation.

    This is the **only** evaluation interface that problems and search
    strategies interact with.  Even a single evaluator is wrapped here.

    Parameters
    ----------
    evaluators : dict[str, BaseEvaluator]
        Named evaluator instances.
    weights : dict[str, float]
        Per-evaluator aggregation weights.
    aggregation : AggregationStrategy
        How to combine per-evaluator scores.
    num_layers : int
        Validated at construction — all evaluators must agree.
    mixer_names : frozenset[str]
        Validated at construction — all evaluators must agree.
    """

    def __init__(
        self,
        evaluators: dict[str, BaseEvaluator[ParametersT]],
        weights: dict[str, float],
        aggregation: AggregationStrategy,
        num_layers: int,
        mixer_names: frozenset[str],
    ) -> None:
        self.evaluators = evaluators
        self.weights = weights
        self.aggregation = aggregation
        self._num_layers = num_layers
        self._mixer_names = mixer_names

    # -- Model structure (validated, not lazy) --------------------------------

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def mixer_names(self) -> frozenset[str]:
        return self._mixer_names

    @property
    def num_mixers(self) -> int:
        return len(self._mixer_names)

    # -- Evaluation -----------------------------------------------------------

    async def evaluate(self, parameters: ParametersT) -> dict[str, float]:
        """Evaluate *parameters* using all evaluators in parallel.

        Returns
        -------
        dict[str, float]
            Evaluator name → score.
        """
        eval_coros = [
            evaluator.evaluate(parameters)
            for evaluator in self.evaluators.values()
        ]
        score_list = await asyncio.gather(*eval_coros)
        return dict(zip(self.evaluators.keys(), score_list))

    async def evaluate_batch(
        self, parameters_list: list[ParametersT]
    ) -> list[dict[str, float]]:
        """Evaluate multiple parameter settings."""
        return [await self.evaluate(p) for p in parameters_list]

    # -- Aggregation (ONLY lives here, not on BaseEvaluator) ------------------

    def aggregate(self, scores: dict[str, float]) -> float:
        """Aggregate per-evaluator *scores* to a single scalar."""
        if self.aggregation == AggregationStrategy.WEIGHTED_AVERAGE:
            return sum(self.weights[name] * score for name, score in scores.items())
        elif self.aggregation == AggregationStrategy.FIRST_ONLY:
            first_name = next(iter(self.evaluators.keys()))
            return scores[first_name]
        elif self.aggregation == AggregationStrategy.MIN:
            return min(scores.values())
        elif self.aggregation == AggregationStrategy.MAX:
            return max(scores.values())
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation}")

    # -- Lifecycle ------------------------------------------------------------

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass  # Lifecycle managed by EvaluatorsConfig via AsyncExitStack


# ---------------------------------------------------------------------------
# EvaluatorsConfig
# ---------------------------------------------------------------------------


class EvaluatorsConfig(BaseModel):
    """Configuration for 1+ evaluators with aggregation strategy.

    Usage::

        async with cfg.create_evaluators(output_dir, backend=backend) as evals:
            scores = await evals.evaluate(placement)
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
            if not (0.99 <= total <= 1.01):
                raise ValueError(
                    f"Weights must sum to 1.0 for weighted_average, got {total:.4f}"
                )

        # Check that all weights are non-negative
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("All weights must be non-negative")

        return self

    @asynccontextmanager
    async def create_evaluators(
        self,
        output_dir: Path,
        backend: "AsyncBackend | SyncBackend | None" = None,
    ) -> AsyncGenerator[Evaluators, None]:
        """Create all evaluators and manage their lifecycle.

        Each evaluator gets its own output subdirectory:
        ``{output_dir}/evals/{name}/``.

        Parameters
        ----------
        output_dir : Path
            Base output directory for all evaluators.
        backend : AsyncBackend | SyncBackend | None
            Pre-created backend instance (lifecycle managed by caller).
            Required for evaluators that need a backend.
            ``None`` for mock evaluators.

        Yields
        ------
        Evaluators
            Container with validated structural properties.
        """
        from place_layers.evaluator.backend_evaluator import BackendEvaluatorConfig

        async with AsyncExitStack() as stack:
            evals_dir = output_dir / "evals"
            evals_dir.mkdir(parents=True, exist_ok=True)

            # Create all evaluators
            evaluators: dict[str, BaseEvaluator] = {}
            for name, config in self.configs.items():
                eval_dir = evals_dir / name
                eval_dir.mkdir(parents=True, exist_ok=True)

                if isinstance(config, BackendEvaluatorConfig):
                    if backend is None:
                        raise ValueError(
                            f"Evaluator '{name}' requires a backend but none "
                            f"was provided. Pass a backend to create_evaluators()."
                        )
                    evaluator = config.create_evaluator(
                        output_dir=eval_dir, backend=backend
                    )
                else:
                    evaluator = config.create_evaluator(output_dir=eval_dir)

                evaluators[name] = await stack.enter_async_context(evaluator)

            # Cross-evaluator validation: all must agree on model structure
            num_layers: int | None = None
            mixer_names: frozenset[str] | None = None

            for name, e in evaluators.items():
                if not hasattr(e, "num_layers"):
                    continue
                if num_layers is None:
                    num_layers = e.num_layers
                    mixer_names = e.mixer_names
                else:
                    if e.num_layers != num_layers:
                        raise ValueError(
                            f"Evaluator num_layers mismatch: "
                            f"{num_layers} vs {e.num_layers}"
                        )
                    if e.mixer_names != mixer_names:
                        raise ValueError(
                            f"Evaluator mixer_names mismatch: "
                            f"{mixer_names} vs {e.mixer_names}"
                        )

            if num_layers is None:
                # Fallback for evaluators that don't expose num_layers
                first = next(iter(evaluators.values()))
                num_layers = first.num_layers
                mixer_names = first.mixer_names

            assert mixer_names is not None

            logger.info(
                "Created %d evaluator(s): %s (%d layers, %d mixers)",
                len(evaluators),
                list(evaluators.keys()),
                num_layers,
                len(mixer_names),
            )

            yield Evaluators(
                evaluators=evaluators,
                weights=self.weights,
                aggregation=self.aggregation,
                num_layers=num_layers,
                mixer_names=mixer_names,
            )
