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
from typing import Any, ClassVar, Literal

from hydra.core.hydra_config import HydraConfig
from pydantic import BaseModel, Field

from place_layers.backend import BackendConfig
from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.command.placement_utils import resolve_placement
from place_layers.evaluator.evaluators import EvaluatorsConfig
from place_layers.landscape.config import objective_result_path
from place_layers.result_handler import ResultHandlerConfig

logger = logging.getLogger(__name__)


class EvaluationResult(BaseModel):
    """Result of evaluating a placement."""

    model_config = {"frozen": True}

    timestamp: str = Field(description="ISO timestamp of evaluation")
    placement: list[str] | None = Field(
        description="Mixer names per layer (null if no placement)"
    )
    placement_indices: list[int] | None = Field(
        description="Mixer indices per layer (null if no placement)"
    )
    objective_name: str = Field(description="Name of the objective")
    score: float = Field(description="Objective score")
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

        backend_cm = self.config.backend.create_backend() if self.config.backend else nullcontext()
        async with backend_cm as backend:
            async with self.config.evaluators.create_evaluators(
                output_dir, backend=backend
            ) as evals:
                resolved = resolve_placement(self.config.placement, evals)

                obj_scores = await evals.evaluate_objectives(resolved.placement)

                for obj_name, obj_score in obj_scores.items():
                    logger.info(f"  {obj_name}: {obj_score:.6f}")

                # Best-effort: compute per-filter metrics from evaluator output dirs
                filter_problem_accuracy: dict[str, float] | None = None
                filter_sample_accuracy: dict[str, float] | None = None
                extra: dict[str, Any] | None = None
                try:
                    from place_layers.evaluator.traces.writer import (
                        summarize_generations_by_filter,
                    )

                    evals_dir = output_dir / "evals"
                    all_summaries: dict[str, dict] = {}
                    for name in evals.evaluators:
                        eval_dir = evals_dir / name
                        # Walk placement subdirectories (e.g. evals/<task>/a48/)
                        # since TraceWriter writes into placement-indexed subdirs
                        trace_dirs = sorted(eval_dir.glob("*/validation.jsonl"))
                        if not trace_dirs:
                            # Fallback: try directly in eval_dir
                            trace_dirs = [eval_dir / "validation.jsonl"]
                        for validation_path in trace_dirs:
                            trace_dir = validation_path.parent
                            try:
                                summaries = summarize_generations_by_filter(trace_dir)
                                for fname, s in summaries.items():
                                    key = f"{name}/{fname}" if len(evals.evaluators) > 1 else fname
                                    all_summaries[key] = s
                            except FileNotFoundError:
                                pass

                    if all_summaries:
                        filter_problem_accuracy = {
                            k: s.problem_accuracy for k, s in all_summaries.items()
                        }
                        filter_sample_accuracy = {
                            k: s.sample_accuracy for k, s in all_summaries.items()
                        }
                        extra = {
                            "filter_task_stats": {
                                k: {
                                    task_name: {
                                        "problem_accuracy": stats.problem_accuracy,
                                        "sample_accuracy": stats.sample_accuracy,
                                        "num_problems": stats.num_problems,
                                        "num_samples": stats.num_samples,
                                    }
                                    for task_name, stats in s.task_stats.items()
                                }
                                for k, s in all_summaries.items()
                            }
                        }
                except Exception as e:
                    logger.warning(f"Failed to compute per-filter results: {e}")

                # Save per-objective results
                result_handler = self.config.result_handler.create_result_handler()
                for obj_name in evals.objective_names:
                    result = EvaluationResult(
                        timestamp=datetime.now().isoformat(),
                        placement=resolved.placement_names,
                        placement_indices=resolved.placement_indices,
                        objective_name=obj_name,
                        score=obj_scores[obj_name],
                        filter_problem_accuracy=filter_problem_accuracy,
                        filter_sample_accuracy=filter_sample_accuracy,
                        extra=extra,
                    )
                    result_path = objective_result_path(
                        output_dir, "evaluation_results", obj_name
                    )
                    await result_handler.save(result, result_path)

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
