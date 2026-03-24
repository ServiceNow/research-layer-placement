"""Primary evaluation interface: wraps 1+ evaluators with an objective.

``Evaluators`` is the **only** evaluation interface that problems and search
strategies interact with.  Even a single evaluator is wrapped in this
container.  Individual ``BaseEvaluator`` subclasses stay clean — evaluate one
thing, return a float, know nothing about objectives.

Configuration
-------------
.. code-block:: yaml

    evaluators:
      configs:
        gsm8k:
          evaluator_type: generative
          tasks: ["gsm8k"]
        mmlu:
          evaluator_type: likelihood
          tasks: ["mmlu"]
      objective:
        objective_type: weighted_average
        weights:
          gsm8k: 0.6
          mmlu: 0.4

Code usage::

    async with config.evaluators.create_evaluators(output_dir, backend=backend) as evals:
        scores = await evals.evaluate(placement)             # dict[str, float]
        obj_scores = evals.objectives(scores)                # dict[str, float]
        obj_scores = await evals.evaluate_objectives(placement)  # combined
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator, Iterable, Self, assert_never

from pydantic import BaseModel, Field, model_validator

from place_layers.evaluator import EvaluatorConfig
from place_layers.evaluator.objective import (
    AggregatingObjectiveConfig,
    BaseObjective,
    IndependentObjectiveConfig,
    ObjectiveConfig,
    WeightedAverageObjectiveConfig,
)

if TYPE_CHECKING:
    from place_layers.backend.base import AsyncBackend, SyncBackend
    from place_layers.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluators (runtime container)
# ---------------------------------------------------------------------------


class Evaluators[ParametersT]:
    """Primary evaluation interface.  Wraps 1+ evaluators with an objective.

    This is the **only** evaluation interface that problems and search
    strategies interact with.  Even a single evaluator is wrapped here.

    Parameters
    ----------
    evaluators : dict[str, BaseEvaluator]
        Named evaluator instances.
    num_layers : int
        Validated at construction — all evaluators must agree.
    mixer_names : frozenset[str]
        Validated at construction — all evaluators must agree.
    objective : BaseObjective
        Objective mapping evaluator scores to optimization objectives.
    """

    def __init__(
        self,
        evaluators: dict[str, BaseEvaluator[ParametersT]],
        num_layers: int,
        mixer_names: frozenset[str],
        objective: BaseObjective,
    ) -> None:
        self.evaluators = evaluators
        self._num_layers = num_layers
        self._mixer_names = mixer_names
        self.objective = objective

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

    # -- Objective names (exposed for search strategies) ----------------------

    @property
    def objective_names(self) -> tuple[str, ...]:
        """Objective names exposed for search strategies to iterate over."""
        return self.objective.objective_names

    # -- Primary evaluation interface (what search strategies call) -----------

    async def evaluate_objectives(
        self,
        parameters: ParametersT,
        names: Iterable[str] | None = None,
    ) -> dict[str, float]:
        """Evaluate and return objective scores.  This is the primary search API.

        Args:
            parameters: Placement to evaluate.
            names: Which objectives to compute.  None = all objectives.

        Internally determines which evaluators are needed (via the objective's
        ``required_evaluators()``), runs only those, and maps through the
        objective.
        """
        if names is None:
            requested = self.objective.objective_names
        else:
            requested = tuple(names)

        # Determine which evaluators are needed
        needed: set[str] = set()
        for name in requested:
            needed |= self.objective.required_evaluators(name)

        # Run only needed evaluators
        eval_names = sorted(needed)
        coros = [self.evaluators[n].evaluate(parameters) for n in eval_names]
        score_list = await asyncio.gather(*coros)
        evaluator_scores = dict(zip(eval_names, score_list))

        # Map through objective, filter to requested names
        all_obj_scores = self.objective.objectives(evaluator_scores)
        return {name: all_obj_scores[name] for name in requested}

    # -- Score mapping (pure function on evaluator scores) --------------------

    def objectives(self, evaluator_scores: dict[str, float]) -> dict[str, float]:
        """Map evaluator scores to objective scores.  Pure function, no I/O.

        Used when evaluator scores are already cached (e.g., BeamEntry.scores,
        ExplorationDataset).  Search strategies store evaluator-level scores
        and compute objective scores on demand via this method.
        """
        return self.objective.objectives(evaluator_scores)

    # -- Low-level: evaluate all evaluators (returns evaluator scores) --------

    async def evaluate(self, parameters: ParametersT) -> dict[str, float]:
        """Evaluate *parameters* using all evaluators in parallel.

        Returns
        -------
        dict[str, float]
            Evaluator name -> score.
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



# ---------------------------------------------------------------------------
# EvaluatorsConfig
# ---------------------------------------------------------------------------


class EvaluatorsConfig(BaseModel):
    """Configuration for 1+ evaluators with an objective.

    Usage::

        async with cfg.create_evaluators(output_dir, backend=backend) as evals:
            scores = await evals.evaluate(placement)
    """

    model_config = {"frozen": True}

    configs: dict[str, EvaluatorConfig] = Field(
        description="Dict of named evaluator configurations"
    )

    objective: ObjectiveConfig = Field(
        description="Objective function mapping evaluator scores to optimization objectives",
    )

    @property
    def objective_names(self) -> tuple[str, ...]:
        """Derive objective names from pure config data — no object creation.

        Mirrors the logic in the runtime objective classes but reads only
        from config fields so it can be used before ``create_evaluators()``.
        """
        evaluator_names = tuple(self.configs.keys())
        match self.objective:
            case IndependentObjectiveConfig():
                return evaluator_names
            case AggregatingObjectiveConfig(include_individual=include):
                if include:
                    return (self.objective.objective_type,) + evaluator_names
                return (self.objective.objective_type,)
            case other:
                assert_never(other)

    @model_validator(mode="after")
    def validate_objective(self) -> Self:
        """Validate objective configuration against evaluator configs."""
        if not self.configs:
            raise ValueError("Must provide at least one evaluator")

        if isinstance(self.objective, WeightedAverageObjectiveConfig):
            if self.objective.weights:
                # Validate weight keys match evaluator keys
                if set(self.objective.weights.keys()) != set(self.configs.keys()):
                    raise ValueError(
                        f"Weight keys {set(self.objective.weights.keys())} must match "
                        f"evaluator keys {set(self.configs.keys())}"
                    )
                # Validate weights sum to 1.0
                total = sum(self.objective.weights.values())
                if not (0.99 <= total <= 1.01):
                    raise ValueError(
                        f"Weights must sum to 1.0 for weighted_average, got {total:.4f}"
                    )
                # Validate non-negative
                if any(w < 0 for w in self.objective.weights.values()):
                    raise ValueError("All weights must be non-negative")
            # Empty weights auto-filled at create_objective() time

        if (
            isinstance(self.objective, AggregatingObjectiveConfig)
            and self.objective.include_individual
            and self.objective.objective_type in self.configs
        ):
            raise ValueError(
                f"Evaluator name '{self.objective.objective_type}' collides with "
                f"aggregate objective type. Rename the evaluator."
            )

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

            # Create all evaluators
            evaluators: dict[str, BaseEvaluator] = {}
            for name, config in self.configs.items():
                eval_dir = evals_dir / name

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
                try:
                    e_num_layers = e.num_layers
                    e_mixer_names = e.mixer_names
                except (AttributeError, RuntimeError):
                    continue
                if num_layers is None:
                    num_layers = e_num_layers
                    mixer_names = e_mixer_names
                else:
                    if e_num_layers != num_layers:
                        raise ValueError(
                            f"Evaluator num_layers mismatch: "
                            f"{num_layers} vs {e_num_layers}"
                        )
                    if e_mixer_names != mixer_names:
                        raise ValueError(
                            f"Evaluator mixer_names mismatch: "
                            f"{mixer_names} vs {e_mixer_names}"
                        )

            if num_layers is not None:
                assert mixer_names is not None
                logger.info(
                    "Created %d evaluator(s): %s (%d layers, %d mixers)",
                    len(evaluators),
                    list(evaluators.keys()),
                    num_layers,
                    len(mixer_names),
                )
            else:
                logger.info(
                    "Created %d evaluator(s): %s (model structure not available)",
                    len(evaluators),
                    list(evaluators.keys()),
                )

            # Create objective from config
            objective = self.objective.create_objective(
                evaluator_names=tuple(evaluators.keys())
            )

            yield Evaluators(
                evaluators=evaluators,
                objective=objective,
                num_layers=num_layers,
                mixer_names=mixer_names,
            )
