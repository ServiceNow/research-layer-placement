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
from place_layers.command.placement_utils import resolve_placement
from place_layers.core.memory_logging import log_memory, reset_peak_memory_stats
from place_layers.evaluator.trace_precompute import TracePrecomputeEvaluatorConfig

logger = logging.getLogger(__name__)


class PrecomputeTracesCommand(BaseCommand):
    """Pre-compute teacher traces using TracePrecomputeEvaluator."""

    config: "PrecomputeTracesCommandConfig"

    async def run(self) -> None:
        """Execute trace pre-computation with streaming and resumption."""
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)
        logger.info(f"Output directory: {output_dir}")

        log_memory("INIT")
        reset_peak_memory_stats()

        async with self.config.evaluator.create_evaluator(output_dir) as evaluator:
            # Resolve placement specification
            resolved = resolve_placement(self.config.placement, evaluator)

            # Run trace generation
            accuracy = await evaluator.evaluate(resolved.placement)
            logger.info(f"Teacher accuracy: {accuracy:.1%}")

        logger.info("Trace precomputation complete!")


class PrecomputeTracesCommandConfig(BaseCommandConfig):
    """Configuration for trace pre-computation command."""

    _command_class: ClassVar[type[BaseCommand]] = PrecomputeTracesCommand

    command: Literal["precompute_traces"] = "precompute_traces"

    evaluator: TracePrecomputeEvaluatorConfig = Field(
        description="Trace precompute evaluator configuration"
    )

    placement: list[str] | None = Field(
        description="Mixer names per layer. If null, uses model's default placement.",
    )
