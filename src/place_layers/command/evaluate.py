"""Evaluate command for running evaluation without search.

This command evaluates a single placement using a configured evaluator,
without running any optimization search. Useful for:
- Testing evaluator setup
- Evaluating baseline placements (e.g., all-attention)
- Comparing specific placements (use Hydra --multirun for sweeps)

Usage
-----
    # Single placement
    python -m place_layers command=evaluate placement=all_attention

    # Sweep over placements
    python -m place_layers command=evaluate placement=all_attention,every_4th_gdn --multirun

    # No placement (for non-supernet models)
    python -m place_layers command=evaluate placement=null
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

from hydra.core.hydra_config import HydraConfig
from pydantic import BaseModel, Field

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.command.placement_utils import resolve_placement
from place_layers.evaluator import EvaluatorConfig
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
    score: float = Field(description="Evaluation score")
    filter_problem_accuracy: dict[str, float] | None = Field(
        default=None,
        description="Per-filter pass@k accuracy (problem-level). Only populated for "
        "generation-based evaluators.",
    )
    filter_sample_accuracy: dict[str, float] | None = Field(
        default=None,
        description="Per-filter sample accuracy (sample-level). Only populated for "
        "generation-based evaluators.",
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Additional evaluator-specific details (best-effort).",
    )


class EvaluateCommand(BaseCommand):
    """Run evaluation on a single placement without search."""

    config: "EvaluateCommandConfig"

    async def run(self) -> None:
        """Execute evaluation."""
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)

        logger.info(f"Output directory: {output_dir}")
        logger.info("Starting evaluation...")

        async with self.config.evaluator.create_evaluator(output_dir) as evaluator:
            # Resolve placement specification
            resolved = resolve_placement(self.config.placement, evaluator)

            # Evaluate
            score = await evaluator.evaluate(resolved.placement)
            logger.info(f"Score: {score:.6f}")

            # Best-effort: compute per-filter metrics for generation-based runs
            filter_problem_accuracy: dict[str, float] | None = None
            filter_sample_accuracy: dict[str, float] | None = None
            extra: dict[str, Any] | None = None
            try:
                from place_layers.evaluator.traces.writer import (
                    summarize_generations_by_filter,
                )

                run_output_dir = getattr(evaluator, "output_dir", None)
                if isinstance(run_output_dir, Path):
                    summaries = summarize_generations_by_filter(run_output_dir)
                    filter_problem_accuracy = {
                        name: s.problem_accuracy for name, s in summaries.items()
                    }
                    filter_sample_accuracy = {
                        name: s.sample_accuracy for name, s in summaries.items()
                    }
                    extra = {
                        "filter_task_stats": {
                            name: {
                                task_name: {
                                    "problem_accuracy": stats.problem_accuracy,
                                    "sample_accuracy": stats.sample_accuracy,
                                    "num_problems": stats.num_problems,
                                    "num_samples": stats.num_samples,
                                }
                                for task_name, stats in s.task_stats.items()
                            }
                            for name, s in summaries.items()
                        }
                    }
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"Failed to compute per-filter results: {e}")

            # Build result
            result = EvaluationResult(
                timestamp=datetime.now().isoformat(),
                evaluator_type=self.config.evaluator.evaluator_type,
                placement=resolved.placement_names,
                placement_indices=resolved.placement_indices,
                score=score,
                filter_problem_accuracy=filter_problem_accuracy,
                filter_sample_accuracy=filter_sample_accuracy,
                extra=extra,
            )

        # Save using result handler
        result_handler = self.config.result_handler.create_result_handler()
        await result_handler.save(result, output_dir / "evaluation_results.json")

        logger.info("Evaluation complete!")


class EvaluateCommandConfig(BaseCommandConfig):
    """Configuration for evaluate command."""

    _impl_class: ClassVar[type[BaseCommand]] = EvaluateCommand

    command: Literal["evaluate"] = "evaluate"

    evaluator: EvaluatorConfig = Field(description="Evaluator configuration")

    placement: list[str] | None = Field(
        description="Layer placement as list of mixer names. Extended cyclically "
        "if shorter than num_layers. Use null for non-supernet models.",
    )

    result_handler: ResultHandlerConfig = Field(
        description="Result handler configuration",
    )
