"""Generative evaluator for direct accuracy measurement.

This evaluator measures task accuracy by actually generating completions
from the model (as opposed to importance-weighted estimation from traces).

Key design principles:
- validation.jsonl is the single source of truth
- Resumable: scans completed problems from disk, skips them
- Lean: only stores Generation records, computes metrics on demand
- Summary computed via summarize_generations() at end

Usage
-----
    config = GenerativeEvaluatorConfig(
        backend=TransformersBackendConfig(model_path="/path/to/model", ...),
        tasks=["gsm8k"],
        ...
    )
    evaluator = config.create_evaluator()
    score = await evaluator.evaluate(placement)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from place_layers.clients.types import GenerationResult
from place_layers.evaluator.base_evaluator import BaseEvaluator
from place_layers.evaluator.base_generative_evaluator import (
    BaseGenerativeEvaluator,
    BaseGenerativeEvaluatorConfig,
)
from place_layers.evaluator.lm_eval_utils import check_correctness_all_filters
from place_layers.evaluator.traces.types import Generation

if TYPE_CHECKING:
    from lm_eval.api.task import Task

logger = logging.getLogger(__name__)


class GenerativeEvaluator(BaseGenerativeEvaluator):
    """Evaluator that measures accuracy through direct generation.

    Supports both async (vLLM) and sync (Transformers) backends, dispatching
    to the appropriate execution path based on backend type.

    Key features:
    - Resumable: uses _prepare_task() which checks completed problems
    - Lean: only stores Generation records, metrics computed at end
    - Progress logging without in-memory accuracy tracking

    Must be used as an async context manager:

        async with config.create_evaluator() as evaluator:
            score = await evaluator.evaluate(placement)
    """

    config: "GenerativeEvaluatorConfig"

    def __init__(
        self, config: "GenerativeEvaluatorConfig", *, output_dir: "Path"
    ) -> None:
        super().__init__(config, output_dir=output_dir)

    def _make_records(
        self, task: "Task", doc: dict, idx: int, result: GenerationResult
    ) -> list[Generation]:
        """Create Generation records from a generation result.

        Truncated sample handling depends on evaluate_truncated_samples config:
        - When False (default): truncated samples marked incorrect (reasoning tasks)
        - When True: truncated samples evaluated for correctness (extraction tasks)
        """
        task_name = task.config.task or ""
        generations = []
        problem_id = f"{task_name}:{idx}"

        for sample_idx, sample in enumerate(result.samples):
            if sample.truncated and not self.config.evaluate_truncated_samples:
                # Truncated and we don't evaluate truncated -> mark incorrect
                logger.debug(f"Truncated sample {problem_id}:{sample_idx}")
                correct = False
                correct_by_filter = None
                filtered_completion_text_by_filter = None
                primary_filter = None
            else:
                filter_results = check_correctness_all_filters(task, doc, sample.text)
                # Choose a primary filter for the scalar `correct` field.
                # Prefer strict extraction when present.
                primary_filter = (
                    "strict-match"
                    if "strict-match" in filter_results
                    else sorted(filter_results.keys())[0]
                )
                correct = filter_results[primary_filter][0]
                correct_by_filter = {k: v[0] for k, v in filter_results.items()}
                filtered_completion_text_by_filter = {
                    k: v[1] for k, v in filter_results.items()
                }

            gen = Generation(
                prompt_tokens=tuple(result.prompt_token_ids),
                completion_tokens=tuple(sample.token_ids),
                correct=correct,
                correct_by_filter=correct_by_filter,
                filtered_completion_text_by_filter=filtered_completion_text_by_filter,
                primary_filter=primary_filter,
                truncated=sample.truncated,
                problem_id=problem_id,
                task_name=task_name,
                sample_idx=sample_idx,
                completion_text=sample.text,
            )
            generations.append(gen)

        return generations


class GenerativeEvaluatorConfig(BaseGenerativeEvaluatorConfig):
    _impl_class: ClassVar[type[BaseEvaluator]] = GenerativeEvaluator

    evaluator_type: Literal["generative"] = "generative"
