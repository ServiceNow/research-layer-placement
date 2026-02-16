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

import asyncio
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
from place_layers.evaluator.multi_evaluator_config import EvaluatorsConfig
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
    auxiliary_scores: dict[str, float] | None = Field(
        default=None,
        description="Scores from auxiliary evaluators (if multi-evaluator mode)",
    )
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

        async with self.config.evaluators.create_evaluators(
            output_dir
        ) as evaluator_instances:
            # Use first evaluator to resolve placement
            first_evaluator = next(iter(evaluator_instances.evaluators.values()))
            resolved = resolve_placement(self.config.placement, first_evaluator)

            # Evaluate with all evaluators in parallel
            logger.info(
                f"Evaluating with {len(evaluator_instances.evaluators)} "
                f"evaluator(s) in parallel..."
            )
            evaluator_tasks = [
                evaluator.evaluate(resolved.placement)
                for evaluator in evaluator_instances.evaluators.values()
            ]
            scores = await asyncio.gather(*evaluator_tasks)

            individual_scores = dict(
                zip(evaluator_instances.evaluators.keys(), scores)
            )
            for name, score in individual_scores.items():
                logger.info(f"  {name}: {score:.6f}")

            # Aggregate scores
            aggregated_score = evaluator_instances._aggregate_scores(individual_scores)
            if len(evaluator_instances.evaluators) > 1:
                logger.info(
                    f"Aggregated score ({self.config.evaluators.aggregation.value}): "
                    f"{aggregated_score:.6f}"
                )

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
            evaluator_type = (
                f"multi[{self.config.evaluators.aggregation.value}]"
                if len(evaluator_instances.evaluators) > 1
                else next(iter(self.config.evaluators.evaluators_dict.values())).evaluator_type
            )
            result = EvaluationResult(
                timestamp=datetime.now().isoformat(),
                evaluator_type=evaluator_type,
                placement=resolved.placement_names,
                placement_indices=resolved.placement_indices,
                score=aggregated_score,
                auxiliary_scores=individual_scores
                if len(individual_scores) > 1
                else None,
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
