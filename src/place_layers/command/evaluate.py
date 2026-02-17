"""Evaluate command for running evaluation without search.

This command evaluates a single placement using a configured evaluator,
without running any optimization search. Useful for:
- Testing evaluator setup
- Evaluating baseline placements (e.g., all-attention)
- Comparing specific placements (use Hydra --multirun for sweeps)
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Literal

from hydra.core.hydra_config import HydraConfig
from pydantic import BaseModel, Field

from place_layers.backend import BackendConfig
from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.command.placement_utils import resolve_placement
from place_layers.evaluator.evaluators import EvaluatorsConfig
from place_layers.result_handler import ResultHandlerConfig

logger = logging.getLogger(__name__)


class EvaluationResult(BaseModel):
    """Result of evaluating a placement."""

    model_config = {"frozen": True}

    timestamp: str = Field(description="ISO timestamp of evaluation")
    evaluator_type: str = Field(description="Type of evaluator used")
    placement: list[str] | None = Field(
        description="Mixer names per layer (null if no placement)"
    )
    placement_indices: list[int] | None = Field(
        description="Mixer indices per layer (null if no placement)"
    )
    scores: dict[str, float] = Field(description="Per-evaluator scores")


class EvaluateCommand(BaseCommand):
    """Run evaluation on a single placement without search."""

    config: "EvaluateCommandConfig"

    async def run(self) -> None:
        """Execute evaluation."""
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)

        logger.info(f"Output directory: {output_dir}")
        logger.info("Starting evaluation...")

        backend_cm = self.config.backend.create_backend() if self.config.backend else nullcontext()
        async with backend_cm as backend:
            async with self.config.evaluators.create_evaluators(
                output_dir, backend=backend
            ) as evals:
                resolved = resolve_placement(self.config.placement, evals)

                # Evaluate with all evaluators in parallel
                logger.info(
                    f"Evaluating with {len(evals.evaluators)} "
                    f"evaluator(s) in parallel..."
                )
                scores = await evals.evaluate(resolved.placement)

                for name, score in scores.items():
                    logger.info(f"  {name}: {score:.6f}")
                if len(scores) > 1:
                    agg = evals.aggregate(scores)
                    logger.info(
                        f"Aggregated score ({self.config.evaluators.aggregation.value}): "
                        f"{agg:.6f}"
                    )

                # Build result
                evaluator_type = (
                    f"multi[{self.config.evaluators.aggregation.value}]"
                    if len(evals.evaluators) > 1
                    else next(iter(self.config.evaluators.configs.values())).evaluator_type
                )
                result = EvaluationResult(
                    timestamp=datetime.now().isoformat(),
                    evaluator_type=evaluator_type,
                    placement=resolved.placement_names,
                    placement_indices=resolved.placement_indices,
                    scores=scores,
                )

        # Save using result handler
        result_handler = self.config.result_handler.create_result_handler()
        await result_handler.save(result, output_dir / "evaluation_results.json")

        logger.info("Evaluation complete!")


class EvaluateCommandConfig(BaseCommandConfig):
    """Configuration for evaluate command."""

    _impl_class: ClassVar[type[BaseCommand]] = EvaluateCommand

    command: Literal["evaluate"] = "evaluate"

    backend: BackendConfig | None = Field(
        default=None,
        description="Backend configuration. None for mock evaluators.",
    )

    evaluators: EvaluatorsConfig = Field(
        description="Evaluator configuration (single or multiple with aggregation)"
    )

    placement: list[str] | None = Field(
        description="Layer placement as list of mixer names. Extended cyclically "
        "if shorter than num_layers. Use null for non-supernet models.",
    )

    result_handler: ResultHandlerConfig = Field(
        description="Result handler configuration",
    )
