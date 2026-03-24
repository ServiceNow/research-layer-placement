"""Batch evaluate command: evaluate a set of placements from an explore run.

Loads placements from an explore run output directory (via the evaluation log
pooling infrastructure), evaluates each one with the configured evaluators,
and saves all results to a single JSON file.

Intermediate per-placement traces are written to the evaluator output dirs
as usual, so the run is resumable if interrupted.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar, Literal

from hydra.core.hydra_config import HydraConfig
from pydantic import BaseModel, Field

from place_layers.backend import BackendConfig
from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.core.types import LayerPlacement
from place_layers.evaluator.evaluators import EvaluatorsConfig
from place_layers.result_handler import ResultHandlerConfig
from place_layers.surrogate.evaluation_log import PooledEvaluationData

logger = logging.getLogger(__name__)


# ============================================================================
# Result models
# ============================================================================


class PlacementEvalResult(BaseModel):
    """Scores for a single placement across all objectives."""

    model_config = {"frozen": True}

    placement: tuple[str, ...]
    scores: dict[str, float]


class BatchEvaluationResult(BaseModel):
    """Result of batch-evaluating a set of placements."""

    model_config = {"frozen": True}

    timestamp: str
    source_dir: str
    num_placements: int
    objective_names: list[str]
    results: list[PlacementEvalResult]


# ============================================================================
# Command
# ============================================================================


class BatchEvaluateCommand(BaseCommand):
    """Evaluate a list of placements loaded from an explore run."""

    config: "BatchEvaluateCommandConfig"

    async def run(self) -> None:
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)

        logger.info("Output directory: %s", output_dir)

        # --- Load placements from source ---
        source_dir = Path(self.config.source_dir)
        logger.info("Loading placements from: %s", source_dir)

        # Find all evaluation log files in the source directory.
        # Supports both naming conventions:
        #   - evaluations_*.jsonl  (batch_eval / newer explore runs)
        #   - exploration_progress_*.jsonl  (older explore runs)
        data_files: list[Path] = sorted(
            source_dir.glob("evaluations_*.jsonl")
        ) or sorted(
            source_dir.glob("exploration_progress_*.jsonl")
        )

        if not data_files:
            raise FileNotFoundError(
                f"No evaluation data files found in {source_dir}. "
                "Expected evaluations_*.jsonl or exploration_progress_*.jsonl files."
            )

        pooled = PooledEvaluationData.from_jsonl(data_files)
        assignments = pooled.assignments

        if self.config.max_placements is not None:
            assignments = assignments[: self.config.max_placements]

        if self.config.extra_placements:
            assignments = list(assignments) + list(self.config.extra_placements)
            logger.info(
                "Added %d extra placements (%d total)",
                len(self.config.extra_placements),
                len(assignments),
            )

        logger.info("Loaded %d placements to evaluate", len(assignments))

        # --- Evaluate ---
        backend_cm = (
            self.config.backend.create_backend() if self.config.backend else nullcontext()
        )
        async with backend_cm as backend:
            async with self.config.evaluators.create_evaluators(
                output_dir, backend=backend
            ) as evals:
                results: list[PlacementEvalResult] = []

                for i, assignment in enumerate(assignments):
                    placement = LayerPlacement(assignments=assignment)
                    obj_scores = await evals.evaluate_objectives(placement)

                    results.append(
                        PlacementEvalResult(placement=assignment, scores=obj_scores)
                    )

                    logger.info(
                        "[%d/%d] %s",
                        i + 1,
                        len(assignments),
                        "  ".join(f"{k}={v:.4f}" for k, v in obj_scores.items()),
                    )

                batch_result = BatchEvaluationResult(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source_dir=str(source_dir),
                    num_placements=len(results),
                    objective_names=list(evals.objective_names),
                    results=results,
                )

        result_handler = self.config.result_handler.create_result_handler()
        result_path = output_dir / "batch_evaluation_results.json"
        await result_handler.save(batch_result, result_path)
        logger.info("Saved %d results to %s", len(results), result_path)


# ============================================================================
# Config
# ============================================================================


class BatchEvaluateCommandConfig(BaseCommandConfig):
    """Configuration for batch_evaluate command."""

    _impl_class: ClassVar[type[BaseCommand]] = BatchEvaluateCommand

    command: Literal["batch_evaluate"] = "batch_evaluate"

    source_dir: str = Field(
        description=(
            "Path to an explore run output directory. "
            "Must contain exploration_results_*.json or evaluations_*.jsonl files."
        )
    )

    backend: BackendConfig | None = Field(
        default=None,
        description="Backend configuration. None for mock evaluators.",
    )

    evaluators: EvaluatorsConfig = Field(
        description="Evaluator configuration (single or multiple with aggregation)"
    )

    result_handler: ResultHandlerConfig = Field(
        description="Result handler configuration",
    )

    max_placements: int | None = Field(
        default=None,
        description=(
            "Maximum number of placements to evaluate. "
            "None means evaluate all placements found in source_dir."
        ),
    )

    extra_placements: list[tuple[str, ...]] = Field(
        default_factory=list,
        description=(
            "Additional specific placements to evaluate, appended after placements "
            "loaded from source_dir. Each placement is a tuple of mixer names, one "
            "per layer (e.g. ['attention', 'gdn', 'attention', ...])."
        ),
    )
