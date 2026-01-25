"""Base class for evaluators that generate completions.

This module provides BaseGenerativeEvaluator which handles the generation
loop infrastructure for evaluators that sample completions from models:

- Streams requests across all tasks for optimal server utilization
- Async worker pool generation for throughput
- SkippedProblem handling for generation errors
- Progress bars
- Resumability via _prepare_task()

Concrete evaluators (GenerativeEvaluator, TracePrecomputeEvaluator) inherit
from this and provide their record factory function.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Sequence

from pydantic import Field

from place_layers.backend.base import (
    AsyncBackend,
    SyncBackend,
)
from place_layers.backend.types import BackendError, GenerateChatRequest, GenerateTextRequest
from place_layers.clients.types import GenerationResult
from place_layers.core.memory_logging import log_memory
from place_layers.core.types import LayerPlacement
from place_layers.evaluator.lm_eval_utils import build_chat_messages, build_text_prompt
from place_layers.evaluator.progress import create_progress
from place_layers.evaluator.task_based_evaluator import (
    TaskBasedEvaluator,
    TaskBasedEvaluatorConfig,
)
from place_layers.evaluator.traces.types import SkippedProblem, TraceRecord
from place_layers.evaluator.traces.writer import summarize_generations

if TYPE_CHECKING:
    from lm_eval.api.task import Task

    from place_layers.evaluator.traces.writer import GenerationSummary

logger = logging.getLogger(__name__)


class BaseGenerativeEvaluator(TaskBasedEvaluator):
    """Base class for evaluators that generate completions.

    Provides the generation loop infrastructure:
    - _run_tasks_sync(): Batched sync generation with memory logging
    - _run_tasks_async(): Worker pool async generation
    - evaluate(): Handles placement, dispatches to sync/async, computes summary

    Subclasses must implement:
    - _make_records(): Create records from generation results
    - _get_fewshot_seed(): Return seed for few-shot sampling (or None)

    Must be used as an async context manager:

        async with config.create_evaluator() as evaluator:
            score = await evaluator.evaluate(placement)
    """

    config: "BaseGenerativeEvaluatorConfig"

    def __init__(self, config: "BaseGenerativeEvaluatorConfig") -> None:
        super().__init__(config)

    @abstractmethod
    def _make_records(
        self, task: "Task", doc: dict, idx: int, result: GenerationResult
    ) -> Sequence[TraceRecord]:
        """Create records from a generation result.

        Args:
            task: The lm-eval task.
            doc: The problem document.
            idx: Problem index.
            result: Generation result with samples.

        Returns:
            Sequence of records (Trace or Generation). Empty sequence means all
            samples were truncated/invalid.
        """
        pass

    async def evaluate(self, parameters: LayerPlacement | None) -> float:
        """Evaluate by generating completions and checking accuracy.

        Args:
            parameters: Layer placement to evaluate, or None for default config.

        Returns:
            Problem accuracy (fraction with any correct sample).
        """
        if parameters is not None:
            await self._set_placement(parameters)
        else:
            # Use default placement (read from backend in __aenter__)
            self._init_trace_writer()

        # Dispatch to async or sync generation
        if isinstance(self._backend, AsyncBackend):
            await self._run_tasks_async()
        elif isinstance(self._backend, SyncBackend):
            self._run_tasks_sync()
        else:
            raise TypeError(f"Unknown backend type: {type(self._backend)}")

        # Compute summary from disk (single source of truth)
        summary = self._summarize()
        logger.info(f"\n{summary}")

        return summary.problem_accuracy

    def _summarize(self) -> "GenerationSummary":
        """Compute summary from disk. Override for different summary types."""
        return summarize_generations(self._output_dir)

    def _run_tasks_sync(self) -> None:
        """Run all tasks as a unified stream (sync backend).

        Streams requests across all tasks to the backend, processing results
        in order. Uses lazy request generation to overlap doc loading.
        """
        assert isinstance(self._backend, SyncBackend)

        # Prepare all tasks and build lookup tables
        key_to_context: dict[str, tuple["Task", str, int]] = {}
        total_problems = 0

        for task_spec in self.config.tasks:
            prepared = self._prepare_task(task_spec, "generate_until")
            if prepared is None:
                continue

            task, task_name, docs, remaining = prepared
            for idx in remaining:
                key = f"{task_name}:{idx}"
                key_to_context[key] = (task, task_name, idx)
            total_problems += len(remaining)

        if total_problems == 0:
            logger.info("All tasks already complete")
            return

        logger.info(f"Streaming {total_problems} problems across {len(self.config.tasks)} tasks")
        log_memory("GENERATION START")

        # Lazy request generator
        def request_generator() -> Iterator[GenerateChatRequest] | Iterator[GenerateTextRequest]:
            for task_spec in self.config.tasks:
                prepared = self._prepare_task(task_spec, "generate_until")
                if prepared is None:
                    continue

                task, task_name, docs, remaining = prepared

                for idx in remaining:
                    if self.config.use_chat_template:
                        yield GenerateChatRequest(
                            key=f"{task_name}:{idx}",
                            messages=list(build_chat_messages(task, docs[idx], task.config.num_fewshot or 0)),
                            n=self.config.samples_per_problem,
                            max_tokens=self.config.max_new_tokens,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                        )
                    else:
                        yield GenerateTextRequest(
                            key=f"{task_name}:{idx}",
                            prompt=build_text_prompt(task, docs[idx], task.config.num_fewshot or 0),
                            n=self.config.samples_per_problem,
                            max_tokens=self.config.max_new_tokens,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                        )

        # Stream to backend
        if self.config.use_chat_template:
            results_iter = self._backend.generate_chat(request_generator())  # type: ignore[arg-type]
        else:
            results_iter = self._backend.generate_text(request_generator())  # type: ignore[arg-type]

        processed = 0

        with create_progress() as progress:
            progress_task = progress.add_task("[cyan]Generating", total=total_problems)

            for result in results_iter:
                # Look up context from key
                task, task_name, idx = key_to_context[result.request_id]
                doc = self._docs_cache[task_name][idx]

                # Create records
                records = self._make_records(task, doc, idx, result)

                if records:
                    self._trace_writer.append(records)
                else:
                    skipped = SkippedProblem(
                        problem_id=f"{task_name}:{idx}",
                        task_name=task_name,
                        reason="no samples returned",
                    )
                    self._trace_writer.append([skipped])

                processed += 1
                progress.advance(progress_task)

        log_memory("GENERATION END")
        logger.info(f"Generation complete: {processed} problems processed")

    async def _run_tasks_async(self) -> None:
        """Run all tasks as a unified stream for optimal server utilization.

        Streams requests across all tasks to the backend, processing results
        as they arrive. Uses lazy request generation to overlap doc loading
        with request processing.
        """
        assert isinstance(self._backend, AsyncBackend)

        # Prepare all tasks and build lookup tables
        # key -> (task, task_name, idx) for result processing
        key_to_context: dict[str, tuple["Task", str, int]] = {}
        total_problems = 0

        for task_spec in self.config.tasks:
            prepared = self._prepare_task(task_spec, "generate_until")
            if prepared is None:
                continue

            task, task_name, docs, remaining = prepared
            for idx in remaining:
                key = f"{task_name}:{idx}"
                key_to_context[key] = (task, task_name, idx)
            total_problems += len(remaining)

        if total_problems == 0:
            logger.info("All tasks already complete")
            return

        logger.info(f"Streaming {total_problems} problems across {len(self.config.tasks)} tasks")

        # Lazy request generator - yields requests across all tasks
        def request_generator() -> Iterator[GenerateChatRequest] | Iterator[GenerateTextRequest]:
            for task_spec in self.config.tasks:
                prepared = self._prepare_task(task_spec, "generate_until")
                if prepared is None:
                    continue

                task, task_name, docs, remaining = prepared

                for idx in remaining:
                    if self.config.use_chat_template:
                        yield GenerateChatRequest(
                            key=f"{task_name}:{idx}",
                            messages=list(build_chat_messages(task, docs[idx], task.config.num_fewshot or 0)),
                            n=self.config.samples_per_problem,
                            max_tokens=self.config.max_new_tokens,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                        )
                    else:
                        yield GenerateTextRequest(
                            key=f"{task_name}:{idx}",
                            prompt=build_text_prompt(task, docs[idx], task.config.num_fewshot or 0),
                            n=self.config.samples_per_problem,
                            max_tokens=self.config.max_new_tokens,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                        )

        # Stream to backend
        if self.config.use_chat_template:
            results_stream = self._backend.generate_chat(request_generator())  # type: ignore[arg-type]
        else:
            results_stream = self._backend.generate_text(request_generator())  # type: ignore[arg-type]

        processed_count = 0

        with create_progress() as progress:
            progress_task = progress.add_task("[cyan]Generating", total=total_problems)

            # Process results as they arrive (may be out of order)
            async for result in results_stream:
                # Handle backend errors
                if isinstance(result, BackendError):
                    task_name = result.key.rsplit(":", 1)[0]
                    skipped = SkippedProblem(
                        problem_id=result.key,
                        task_name=task_name,
                        reason=f"{result.error_type}: {result.message}",
                    )
                    self._trace_writer.append([skipped])
                    processed_count += 1
                    progress.advance(progress_task)
                    continue

                # Look up context from key
                task, task_name, idx = key_to_context[result.request_id]
                doc = self._docs_cache[task_name][idx]

                # Create records
                records = self._make_records(task, doc, idx, result)

                if records:
                    self._trace_writer.append(records)
                else:
                    skipped = SkippedProblem(
                        problem_id=f"{task_name}:{idx}",
                        task_name=task_name,
                        reason="no samples returned",
                    )
                    self._trace_writer.append([skipped])

                processed_count += 1
                progress.advance(progress_task)

        logger.info(f"Generation complete: {processed_count} problems processed")


class BaseGenerativeEvaluatorConfig(TaskBasedEvaluatorConfig):
    """Generation parameters for GenerativeEvaluator and TracePrecomputeEvaluator."""

    samples_per_problem: int = Field(
        description="Number of completions per problem",
        gt=0,
    )
    max_new_tokens: int = Field(
        description="Maximum tokens to generate per completion",
        gt=0,
    )
    temperature: float = Field(
        description="Sampling temperature (0.0 for greedy)",
        ge=0.0,
    )
    top_p: float = Field(
        description="Nucleus sampling parameter",
        gt=0.0,
        le=1.0,
    )
    use_chat_template: bool = Field(
        description="Whether to use chat template formatting. When False, prompts are "
        "formatted as raw text without any chat structure. "
        "Use False for evaluating base models or when chat template is undesired.",
    )
    evaluate_truncated_samples: bool = Field(
        description="Whether to evaluate truncated samples for correctness. "
        "When True, samples that reach max_new_tokens are still checked against the ground truth. "
        "Useful for extraction tasks where the answer may be complete before hitting the limit. "
        "When False, truncated samples are automatically marked as incorrect.",
    )
