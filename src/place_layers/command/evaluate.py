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
import time
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Literal

from hydra.core.hydra_config import HydraConfig
from pydantic import BaseModel, Field

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.core.types import LayerPlacement
from place_layers.evaluator import EvaluatorConfig
from place_layers.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


def _get_filter_accuracy(evaluator: BaseEvaluator) -> dict[str, "FilterAccuracy"] | None:
    """Extract per-filter accuracy from evaluator's last summary if available."""
    summary = getattr(evaluator, "last_summary", None)
    if summary is None:
        return None

    filter_stats = getattr(summary, "filter_stats", None)
    if not filter_stats:
        return None

    total_problems = summary.total_problems
    total_samples = summary.total_samples

    return {
        filter_name: FilterAccuracy(
            problems_correct=fstats.problems_correct,
            samples_correct=fstats.samples_correct,
            problem_accuracy=fstats.problem_accuracy(total_problems),
            sample_accuracy=fstats.sample_accuracy(total_samples),
        )
        for filter_name, fstats in filter_stats.items()
    }


class FilterAccuracy(BaseModel):
    """Per-filter accuracy stats."""

    model_config = {"frozen": True}

    problems_correct: int = Field(description="Problems with at least one correct sample")
    samples_correct: int = Field(description="Number of correct samples")
    problem_accuracy: float = Field(description="Problem accuracy (pass@k)")
    sample_accuracy: float = Field(description="Sample accuracy")


class EvaluationResult(BaseModel):
    """Result of evaluating a placement."""

    model_config = {"frozen": True}

    placement: list[str] | None = Field(
        description="Mixer names per layer (null if no placement)"
    )
    score: float = Field(description="Evaluation score (accuracy)")
    brier_score: float | None = Field(
        default=None,
        description="Brier score for likelihood evaluation (lower is better, null for non-likelihood)"
    )
    duration_seconds: float = Field(
        description="Total evaluation runtime in seconds (includes data loading)"
    )
    inference_duration_seconds: float = Field(
        description="Model inference time in seconds (excludes data loading)"
    )
    placement_indices: list[int] | None = Field(
        description="Mixer indices per layer (null if no placement)"
    )
    filter_accuracy: dict[str, FilterAccuracy] | None = Field(
        default=None,
        description="Per-filter accuracy stats (null if no filters)"
    )


class EvaluateCommandOutput(BaseModel):
    """Output of the evaluate command."""

    model_config = {"frozen": True}

    timestamp: str = Field(description="ISO timestamp of evaluation")
    evaluator_type: str = Field(description="Type of evaluator used")
    mixer_names: list[str] | None = Field(description="Available mixer types (null if not discovered)")
    num_layers: int | None = Field(description="Number of layers (null if not discovered)")
    result: EvaluationResult = Field(description="Evaluation result")


class EvaluateCommand(BaseCommand):
    """Run evaluation on a single placement without search."""

    config: "EvaluateCommandConfig"

    async def run(self) -> None:
        """Execute evaluation."""
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)

        logger.info(f"Output directory: {output_dir}")
        logger.info("Starting evaluation...")

        # Create and use evaluator with context manager for proper cleanup
        async with self.config.evaluator.create_evaluator() as evaluator:
            # Handle placement
            if self.config.placement is not None:
                # Placement specified - need model structure for validation
                num_layers = evaluator.num_layers
                num_mixers = evaluator.num_mixers
                mixer_names = evaluator.mixer_names

                logger.info(
                    f"Evaluator: {self.config.evaluator.evaluator_type}, "
                    f"{num_layers} layers, {num_mixers} mixers"
                )
                logger.info(f"Available mixers: {mixer_names}")

                # Build name-to-index mapping
                name_to_idx = {name: idx for idx, name in enumerate(mixer_names)}

                # Check if model supports placement (multiple mixer types)
                supports_placement = len(set(mixer_names)) > 1
                if not supports_placement:
                    raise ValueError(
                        f"Placement specified but model only has one mixer type "
                        f"({mixer_names[0]}). Use placement=null for "
                        f"non-supernet models."
                    )

                # Extend layers cyclically if shorter than num_layers (periodic BC)
                spec_layers = self.config.placement
                if len(spec_layers) < num_layers:
                    extended_layers = [
                        spec_layers[i % len(spec_layers)] for i in range(num_layers)
                    ]
                    logger.info(
                        f"Extended {len(spec_layers)} layers to {num_layers} (periodic BC)"
                    )
                elif len(spec_layers) > num_layers:
                    raise ValueError(
                        f"Placement has {len(spec_layers)} layers, "
                        f"but model only has {num_layers} layers"
                    )
                else:
                    extended_layers = list(spec_layers)

                # Convert mixer names to indices
                try:
                    indices = tuple(name_to_idx[name] for name in extended_layers)
                except KeyError as e:
                    raise ValueError(
                        f"Unknown mixer name in placement: {e}. "
                        f"Valid names: {list(name_to_idx.keys())}"
                    ) from e

                # Create LayerPlacement and evaluate
                layer_placement = LayerPlacement(assignments=indices)
                logger.info(f"Evaluating placement: {extended_layers[:6]}...")
                start_time = time.perf_counter()
                score = await evaluator.evaluate(layer_placement)
                total_duration = time.perf_counter() - start_time

                # Get inference duration (excludes data loading)
                inference_duration = getattr(evaluator, "inference_duration", 0.0)

                # Get brier_score if available (likelihood evaluator)
                brier_score = getattr(evaluator, "brier_score", None)

                # Get per-filter accuracy if available (generative evaluator)
                filter_accuracy = _get_filter_accuracy(evaluator)

                result = EvaluationResult(
                    placement=extended_layers,
                    score=score,
                    brier_score=brier_score,
                    duration_seconds=total_duration,
                    inference_duration_seconds=inference_duration,
                    placement_indices=list(indices),
                    filter_accuracy=filter_accuracy,
                )
            else:
                # No placement - evaluate default configuration without model structure
                logger.info("No placement specified, evaluating default configuration")
                num_layers = None
                mixer_names = None

                # Evaluate with None placement (evaluator handles this)
                start_time = time.perf_counter()
                score = await evaluator.evaluate(None)  # type: ignore[arg-type]
                total_duration = time.perf_counter() - start_time

                # Get inference duration (excludes data loading)
                inference_duration = getattr(evaluator, "inference_duration", 0.0)

                # Get brier_score if available (likelihood evaluator)
                brier_score = getattr(evaluator, "brier_score", None)

                # Get per-filter accuracy if available (generative evaluator)
                filter_accuracy = _get_filter_accuracy(evaluator)

                result = EvaluationResult(
                    placement=None,
                    score=score,
                    brier_score=brier_score,
                    duration_seconds=total_duration,
                    inference_duration_seconds=inference_duration,
                    placement_indices=None,
                    filter_accuracy=filter_accuracy,
                )

            logger.info(
                f"Score: {score:.6f} (total: {total_duration:.1f}s, inference: {inference_duration:.1f}s)"
            )

        # Save results
        output = EvaluateCommandOutput(
            timestamp=datetime.now().isoformat(),
            evaluator_type=self.config.evaluator.evaluator_type,
            mixer_names=list(mixer_names) if mixer_names else None,
            num_layers=num_layers,
            result=result,
        )

        output_path = output_dir / "evaluation_results.json"
        with open(output_path, "w") as f:
            f.write(output.model_dump_json(indent=2))

        logger.info(f"Results saved to: {output_path}")
        logger.info("Evaluation complete!")


class EvaluateCommandConfig(BaseCommandConfig):
    """Configuration for evaluate command."""

    _command_class: ClassVar[type[BaseCommand]] = EvaluateCommand

    command: Literal["evaluate"] = "evaluate"

    # Evaluator configuration (discriminated union)
    # The evaluator provides model structure (num_layers, mixer_names) after initialization
    evaluator: EvaluatorConfig = Field(description="Evaluator configuration")

    # Single placement (use Hydra --multirun for sweeps)
    placement: list[str] | None = Field(
        default=None,
        description="Layer placement as list of mixer names. Extended cyclically "
        "if shorter than num_layers (periodic BC). Use null for non-supernet models."
    )
