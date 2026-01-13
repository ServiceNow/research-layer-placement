"""Pre-compute teacher traces for importance-weighted evaluation.

This command generates reasoning traces from a teacher model,
records their correctness (using lm-eval task definitions) and log-probabilities,
and saves them incrementally for use during importance-weighted accuracy estimation.

Usage
-----
    python -m place_layers \\
        command=precompute_traces \\
        evaluator.tasks='[gsm8k]' \\
        evaluator.max_problems_per_task=100 \\
        evaluator.samples_per_problem=8 \\
        evaluator.max_new_tokens=1024

Key parameters (under evaluator.*):
- tasks: List of lm-eval tasks (e.g., '[gsm8k,triviaqa]')
- max_problems_per_task: Limit problems per task (null for all)
- samples_per_problem: Number of traces to sample per problem
- max_new_tokens: Maximum tokens to generate (must be enough for full solutions)
- temperature: Sampling temperature (default 0.7)
- num_fewshot: Number of few-shot examples (default 5)

Architecture
------------
- Uses TracePrecomputeEvaluator for generation (inherits BackendEvaluator)
- lm-eval: Task definitions, prompt formatting, correctness checking
- Streaming: Process problems as they complete, write incrementally
- Resumption: State reconstructed from traces on disk (source of truth)
- Metadata: Config hash validation prevents resuming with different config

Output structure
----------------
    output_dir/
        metadata.json       # Config hash, teacher model, samples_per_problem
        validation.jsonl    # All valid traces (HF-loadable via load_dataset)
        skipped.jsonl       # Skipped problems (for debugging)

The validation.jsonl file can be loaded directly with HuggingFace datasets:

    from datasets import load_dataset
    ds = load_dataset(output_dir, split='validation')
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar, Literal

from hydra.core.hydra_config import HydraConfig
from pydantic import Field

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.core.types import LayerPlacement
from place_layers.core.memory_logging import (
    log_memory,
    reset_peak_memory_stats,
)
from place_layers.evaluator.trace_precompute import TracePrecomputeEvaluatorConfig
from place_layers.evaluator.traces.writer import (
    PrecomputeMetadata,
    compute_config_hash,
    load_metadata,
    save_metadata,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Command Implementation
# =============================================================================


class PrecomputeTracesCommand(BaseCommand):
    """Pre-compute teacher traces using TracePrecomputeEvaluator."""

    config: "PrecomputeTracesCommandConfig"

    async def run(self) -> None:
        """Execute trace pre-computation with streaming and resumption."""
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)
        logger.info(f"Output directory: {output_dir}")

        # Log initial memory state and reset peak tracking
        log_memory("INIT")
        reset_peak_memory_stats()

        # Compute config hash for resumption validation
        config_dict = self.config.model_dump(mode="json")
        config_hash = compute_config_hash(config_dict)

        # Check for existing metadata (resumption case)
        existing_metadata = load_metadata(output_dir)
        if existing_metadata is not None:
            if existing_metadata.config_hash != config_hash:
                raise ValueError(
                    f"Config hash mismatch: existing={existing_metadata.config_hash}, "
                    f"current={config_hash}. Cannot resume with different config."
                )
            logger.info("Resuming from existing traces")
        else:
            # Save metadata for new run
            metadata = PrecomputeMetadata(
                config_hash=config_hash,
                teacher_model=self._get_model_name(),
                samples_per_problem=self.config.evaluator.samples_per_problem,
            )
            save_metadata(output_dir, metadata)
            logger.info("Starting new precomputation run")

        # Create and run evaluator
        async with self.config.evaluator.create_evaluator() as evaluator:
            # Convert placement (if specified) to LayerPlacement
            placement: LayerPlacement | None = None
            if self.config.placement is not None:
                mixer_names = evaluator.mixer_names
                name_to_idx = {name: idx for idx, name in enumerate(mixer_names)}

                # Extend layers cyclically if shorter than num_layers
                num_layers = evaluator.num_layers
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

                # Convert to indices
                try:
                    indices = tuple(name_to_idx[name] for name in extended_layers)
                except KeyError as e:
                    raise ValueError(
                        f"Unknown mixer name in placement: {e}. "
                        f"Valid names: {list(name_to_idx.keys())}"
                    ) from e

                placement = LayerPlacement(assignments=indices)
                logger.info(f"Using placement: {extended_layers[:4]}...")

            # Run trace generation
            accuracy = await evaluator.evaluate(placement)
            logger.info(f"Teacher accuracy: {accuracy:.1%}")

        logger.info("Trace precomputation complete!")

    def _get_model_name(self) -> str:
        """Get model name/path from evaluator's backend config."""
        from place_layers.backend.base import (
            TransformersBackendConfig,
            VLLMBackendConfig,
        )

        backend = self.config.evaluator.backend
        if isinstance(backend, VLLMBackendConfig):
            return backend.model
        elif isinstance(backend, TransformersBackendConfig):
            return backend.model_path
        return "unknown"


# =============================================================================
# Configuration
# =============================================================================


class PrecomputeTracesCommandConfig(BaseCommandConfig):
    """Configuration for trace pre-computation command."""

    _command_class: ClassVar[type[BaseCommand]] = PrecomputeTracesCommand

    command: Literal["precompute_traces"] = "precompute_traces"

    # Evaluator configuration (handles backend, tasks, generation params)
    evaluator: TracePrecomputeEvaluatorConfig = Field(
        description="Trace precompute evaluator configuration"
    )

    # Mixer placement (optional, for supernet models)
    # Kept at command level since it's a runtime parameter, not evaluator config
    placement: list[str] | None = Field(
        default=None,
        description="Mixer names per layer (e.g., ['attention', 'gdn', ...]). "
        "If None, uses model's default placement.",
    )
