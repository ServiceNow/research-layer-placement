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
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

from pydantic import Field

from place_layers.backend.base import (
    AsyncBackend,
    SyncBackend,
)
from place_layers.backend.types import GenerateChatRequest, GenerateTextRequest
from place_layers.clients.types import GenerationResult
from place_layers.core.memory_logging import log_memory
from place_layers.core.types import LayerPlacement
from place_layers.evaluator.lm_eval_utils import build_chat_messages, build_text_prompt
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

    Provides the generation loop infrastructure via TaskBasedEvaluator's
    streaming methods, with generation-specific hooks:
    - _build_request(): Builds GenerateChatRequest or GenerateTextRequest
    - _stream_to_backend_*(): Routes to backend.generate_chat/generate_text
    - _process_result(): Creates records via _make_records() and writes to disk

    Subclasses must implement:
    - _make_records(): Create records from generation results
    - _get_fewshot_seed(): Return seed for few-shot sampling (or None)

    Must be used as an async context manager:

        async with config.create_evaluator() as evaluator:
            score = await evaluator.evaluate(placement)
    """

    config: "BaseGenerativeEvaluatorConfig"

    def __init__(
        self,
        config: "BaseGenerativeEvaluatorConfig",
        *,
        output_dir: Path,
        backend: "AsyncBackend | SyncBackend",
    ) -> None:
        super().__init__(config, output_dir=output_dir, backend=backend)

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

    # -------------------------------------------------------------------------
    # Abstract hook implementations from TaskBasedEvaluator
    # -------------------------------------------------------------------------

    def _expected_output_type(self) -> Literal["generate_until", "multiple_choice"]:
        """Generative evaluators expect 'generate_until' tasks."""
        return "generate_until"

    def _build_request(
        self, task: "Task", doc: dict, idx: int, task_name: str
    ) -> tuple[GenerateChatRequest | GenerateTextRequest, None]:
        """Build generation request. Context is None (not needed for generation).

        Args:
            task: The lm-eval task.
            doc: The problem document.
            idx: Problem index within the task.
            task_name: Name of the task.

        Returns:
            (request, None) - request is GenerateChatRequest or GenerateTextRequest.
        """
        key = f"{task_name}:{idx}"
        if self.config.use_chat_template:
            messages = list(
                build_chat_messages(task, doc, task.config.num_fewshot or 0)
            )
            if self.config.system_instruction:
                messages.insert(
                    0, {"role": "system", "content": self.config.system_instruction}
                )
            return GenerateChatRequest(
                key=key,
                messages=messages,
                n=self.config.samples_per_problem,
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            ), None
        else:
            return GenerateTextRequest(
                key=key,
                prompt=build_text_prompt(task, doc, task.config.num_fewshot or 0),
                n=self.config.samples_per_problem,
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            ), None

    def _stream_to_backend_async(
        self, requests: AsyncIterable[Any]
    ) -> AsyncIterator[Any]:
        """Stream generation requests to async backend.

        Args:
            requests: Async iterable of GenerateChatRequest or GenerateTextRequest.

        Returns:
            Async iterator of GenerationResult or BackendError.
        """
        assert isinstance(self._backend, AsyncBackend)
        if self.config.use_chat_template:
            return self._backend.generate_chat(requests)
        else:
            return self._backend.generate_text(requests)

    def _stream_to_backend_sync(
        self, requests: Iterable[Any]
    ) -> Iterator[Any]:
        """Stream generation requests to sync backend.

        Args:
            requests: Iterable of GenerateChatRequest or GenerateTextRequest.

        Returns:
            Iterator of GenerationResult or BackendError.
        """
        assert isinstance(self._backend, SyncBackend)
        if self.config.use_chat_template:
            return self._backend.generate_chat(requests)
        else:
            return self._backend.generate_text(requests)

    def _process_result(
        self,
        result: GenerationResult,
        context: None,
        task: "Task",
        task_name: str,
        doc: dict,
        idx: int,
    ) -> None:
        """Process generation result by creating records and writing to disk.

        Args:
            result: GenerationResult from backend.
            context: Unused (None for generation).
            task: The lm-eval task.
            task_name: Name of the task.
            doc: The problem document.
            idx: Problem index within the task.
        """
        assert self._trace_writer is not None
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

    # -------------------------------------------------------------------------
    # Main evaluation interface
    # -------------------------------------------------------------------------

    async def evaluate(self, parameters: LayerPlacement | None) -> float:
        """Evaluate by generating completions and checking accuracy.

        Args:
            parameters: Layer placement to evaluate, or None for default config.

        Returns:
            Problem accuracy (fraction with any correct sample).
        """
        if parameters is not None:
            await self._set_placement(parameters)
        self._init_trace_writer()

        log_memory("GENERATION START")

        # Dispatch to base class streaming methods
        if isinstance(self._backend, AsyncBackend):
            await self._run_streaming_async()
        elif isinstance(self._backend, SyncBackend):
            await self._run_streaming_sync()
        else:
            raise TypeError(f"Unknown backend type: {type(self._backend)}")

        log_memory("GENERATION END")

        # Compute summary from disk (single source of truth)
        summary = self._summarize()
        logger.info(f"\n{summary}")

        return summary.problem_accuracy

    def _summarize(self) -> "GenerationSummary":
        """Compute summary from disk. Override for different summary types."""
        return summarize_generations(self._output_dir)


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
    system_instruction: str | None = Field(
        default=None,
        description="System message prepended to chat prompts. "
        "Only used when use_chat_template=True. "
        "Example: 'Answer directly without reasoning.'",
    )
    evaluate_truncated_samples: bool = Field(
        description="Whether to evaluate truncated samples for correctness. "
        "When True, samples that reach max_new_tokens are still checked against the ground truth. "
        "Useful for extraction tasks where the answer may be complete before hitting the limit. "
        "When False, truncated samples are automatically marked as incorrect.",
    )
