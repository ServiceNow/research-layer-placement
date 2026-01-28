"""Likelihood-based evaluator for multiple-choice tasks.

This evaluator measures accuracy on multiple-choice tasks (like MMLU) by
computing log-likelihood of each answer choice and selecting the highest.

Unlike generative evaluation which samples completions and checks correctness,
likelihood evaluation:
- Computes log P(choice | context) for each choice
- Selects argmax as the prediction
- Compares to gold answer index

Scoring Approach: Input Token Logprobs (NOT Generation)
-------------------------------------------------------
This evaluator uses **input token scoring**, not generation. The key insight
is that we don't need the model to generate anything - we just need the
probability it assigns to tokens we provide as input.

For each choice (e.g., "A", "B", "C", "D"), we:

1. **Send the answer as INPUT** (not as a generation target):
   ```
   messages = [
       {role: "user", content: "Question...\\nA. ...\\nB. ...\\nAnswer:"},
       {role: "assistant", content: "B"}  # <-- Answer is INPUT
   ]
   ```

2. **Get logprobs for input tokens** (backend-specific):

   **vLLM backend**: Request `prompt_logprobs` which returns logprobs for all
   INPUT tokens. vLLM requires `max_tokens >= 1`, so we set `max_tokens=1`,
   but we never read the generated token - only `prompt_logprobs`.

   **Transformers backend**: Tokenize the full sequence (context + answer),
   run a single forward pass, compute log_softmax over logits. No generation
   loop - just one forward pass on the complete input.

   Both backends produce logprobs for the full input sequence:
   ```
   [..., <|im_start|>, assistant, \\n, B, <|im_end|>, ...]
                                       ^
                                       └── We extract this logprob
   ```

3. **Use diff-based scoring**: Compare token sequences across all choices to
   find where they differ (the answer token position). This is chat-template
   agnostic - works regardless of how the template wraps messages.

This approach is more principled than generation-based scoring because:
- We measure exactly P(answer | context), not P(answer | context, model's prefix)
- No sampling variance - deterministic logprob extraction
- Faster - no autoregressive generation needed (just one forward pass)

Key design principles:
- Uses build_chat_messages for context construction (shared with generative)
- Uses batch_score_chat with diff-based scoring (template-agnostic)
- Clean separation from generative evaluation (different paradigm)
- Supports both sync (Transformers) and async (vLLM) backends
- Supports length normalization (acc_norm) for fairer comparison
- Resumable via HydraConfig output directory

Usage
-----
    config = LikelihoodEvaluatorConfig(
        backend=TransformersBackendConfig(model_path="/path/to/model", ...),
        tasks=["mmlu_abstract_algebra"],
        num_fewshot=5,
        ...
    )
    evaluator = config.create_evaluator()
    score = await evaluator.evaluate(placement)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, Field

from place_layers.backend import (
    ChatMessage,
    LogProbResult,
    ScoreChatRequest,
    ScoreChatResult,
    ScoreTextRequest,
    ScoreTextResult,
)
from place_layers.backend.base import AsyncBackend, SyncBackend
from place_layers.core.types import LayerPlacement
from place_layers.evaluator.base_evaluator import BaseEvaluator
from place_layers.evaluator.lm_eval_utils import build_chat_messages
from place_layers.evaluator.task_based_evaluator import (
    TaskBasedEvaluator,
    TaskBasedEvaluatorConfig,
)
from place_layers.evaluator.traces.types import (
    ChatLikelihoodRecord,
    TextLikelihoodRecord,
)
from place_layers.evaluator.traces.writer import summarize_likelihood

if TYPE_CHECKING:
    from lm_eval.api.task import Task

logger = logging.getLogger(__name__)


class PredictionResult(BaseModel):
    """Result of computing predictions from log-probabilities."""

    model_config = {"frozen": True}

    log_probs: list[float] = Field(description="Log-probability for each choice")
    pred_idx: int = Field(description="Predicted choice index (argmax of log_probs)")
    correct: bool = Field(description="Whether prediction matches gold answer")
    pred_idx_norm: int = Field(description="Predicted choice index with length normalization")
    correct_norm: bool = Field(description="Whether length-normalized prediction matches gold answer")


class ChatChoiceMetadata(BaseModel):
    """Metadata for chat-based multiple-choice scoring.

    Used when use_chat_template=True.
    """

    model_config = {"frozen": True}

    gold_idx: int = Field(description="Index of the correct answer")
    lengths: list[int] = Field(description="Character length of each choice (for normalization)")
    raw_choices: list[str] = Field(description="Original choice text strings")
    context_messages: list[ChatMessage] = Field(description="Context messages for this problem")
    choice_messages: list[ChatMessage] = Field(description="Choice messages (one per choice)")


class TextChoiceMetadata(BaseModel):
    """Metadata for text-based multiple-choice scoring.

    Used when use_chat_template=False.
    """

    model_config = {"frozen": True}

    gold_idx: int = Field(description="Index of the correct answer")
    lengths: list[int] = Field(description="Character length of each choice (for normalization)")
    raw_choices: list[str] = Field(description="Original choice text strings")
    context: str = Field(description="Context string (few-shot + question)")
    choices: list[str] = Field(description="Choice strings with delimiters")


ChoiceMetadata = ChatChoiceMetadata | TextChoiceMetadata


# =============================================================================
# Helper functions (pure functions, no class dependency)
# =============================================================================


def _get_gold_idx(task: "Task", doc: dict, choices: list[str]) -> int:
    """Extract gold answer index from task and document.

    Handles the various formats that lm-eval tasks use to specify the correct answer:
    - int: Direct index
    - str: Choice text (looked up in choices list)
    - list: Multiple correct answers (uses first one)

    Args:
        task: The lm-eval task.
        doc: The problem document.
        choices: List of choice strings.

    Returns:
        Index of the gold answer in the choices list.
    """
    gold_target = task.doc_to_target(doc)
    if isinstance(gold_target, int):
        return gold_target
    elif isinstance(gold_target, str):
        return choices.index(gold_target)
    else:
        # For multiple correct answers, use first one
        return gold_target[0] if gold_target else 0


def _compute_prediction(
    results: list[LogProbResult],
    gold_idx: int,
    lengths: list[int],
) -> PredictionResult:
    """Compute predictions from log-prob results.

    Takes the log-probabilities from the backend and computes:
    - Raw prediction (argmax of log-probs)
    - Length-normalized prediction (argmax of log-probs / lengths)

    Args:
        results: Log-prob results from backend (one per choice).
        gold_idx: Index of the correct answer.
        lengths: Character lengths of each choice (for normalization).

    Returns:
        PredictionResult with log_probs, predictions, and correctness flags.
    """
    log_probs = [r.total_log_prob for r in results]
    pred_idx = int(np.argmax(log_probs))
    correct = pred_idx == gold_idx

    normalized = np.array(log_probs) / np.array(lengths)
    pred_idx_norm = int(np.argmax(normalized))
    correct_norm = pred_idx_norm == gold_idx

    return PredictionResult(
        log_probs=log_probs,
        pred_idx=pred_idx,
        correct=correct,
        pred_idx_norm=pred_idx_norm,
        correct_norm=correct_norm,
    )


def _build_choice_chats(
    task: "Task", doc: dict
) -> tuple[list[ChatMessage], list[ChatMessage], ChatChoiceMetadata]:
    """Build chat context and completion messages for each choice.

    Uses build_chat_messages for context (few-shot + question), then creates
    a completion message for each choice. The batch_score_chat method
    uses diff-based scoring that is template-agnostic.

    Args:
        task: The lm-eval task.
        doc: The problem document.

    Returns:
        Tuple of (context_msgs, completion_msgs, metadata) where:
        - context_msgs: The shared context messages (system, user, etc.)
        - completion_msgs: List of assistant messages, one per choice
        - metadata: ChatChoiceMetadata with gold_idx, lengths, etc.
    """
    # Get context messages using the shared builder
    context_msgs = build_chat_messages(task, doc, task.config.num_fewshot or 0)

    # Get choices and gold answer
    # doc_to_choice is on ConfigurableTask, but all YAML-defined tasks use it
    choices: list[str] = task.doc_to_choice(doc)  # type: ignore[attr-defined]
    gold_idx = _get_gold_idx(task, doc, choices)

    # Build completion messages (one per choice)
    completion_msgs: list[ChatMessage] = []
    lengths: list[int] = []

    for choice in choices:
        completion_msgs.append({"role": "assistant", "content": choice})
        lengths.append(len(choice))  # Character length for normalization

    metadata = ChatChoiceMetadata(
        gold_idx=gold_idx,
        lengths=lengths,
        raw_choices=choices,
        context_messages=context_msgs,
        choice_messages=completion_msgs,
    )
    return context_msgs, completion_msgs, metadata


def _build_choice_texts(
    task: "Task", doc: dict
) -> tuple[str, list[str], TextChoiceMetadata]:
    """Build raw text context and choices for text-based scoring.

    This method constructs prompts WITHOUT chat template formatting.
    Used when use_chat_template=False.

    Args:
        task: The lm-eval task.
        doc: The problem document.

    Returns:
        Tuple of (context, choices, metadata) where:
        - context: Few-shot examples + question as raw text
        - choices: List of choice strings (continuations with delimiters)
        - metadata: TextChoiceMetadata with gold_idx, lengths, etc.
    """
    # Build context: few-shot examples + question as raw text
    num_fewshot = task.config.num_fewshot or 0
    context: str
    if num_fewshot > 0:
        # fewshot_context returns few-shot examples AND the current question
        fewshot_ctx = task.fewshot_context(doc, num_fewshot)
        context = str(fewshot_ctx) if fewshot_ctx else ""
    else:
        # Zero-shot: just the question
        question_raw = task.doc_to_text(doc)
        context = str(question_raw) if question_raw is not None else ""

    # Get choices and gold answer
    raw_choices: list[str] = task.doc_to_choice(doc)  # type: ignore[attr-defined]
    gold_idx = _get_gold_idx(task, doc, raw_choices)

    # Match lm-eval's _encode_pair approach:
    # 1. Move trailing whitespace from context to continuation
    # 2. Prepend target_delimiter to choices
    # This ensures proper tokenization at word boundaries
    n_spaces = len(context) - len(context.rstrip())
    trailing_ws = context[-n_spaces:] if n_spaces > 0 else ""
    context = context.rstrip()

    delimiter = task.config.target_delimiter
    choices = [trailing_ws + delimiter + c for c in raw_choices]

    # Build lengths for normalization (use raw choices for char-level normalization)
    lengths = [len(choice) for choice in raw_choices]

    metadata = TextChoiceMetadata(
        gold_idx=gold_idx,
        lengths=lengths,
        raw_choices=raw_choices,
        context=context,
        choices=choices,
    )
    return context, choices, metadata


class LikelihoodEvaluator(TaskBasedEvaluator):
    """Evaluator for multiple-choice tasks using log-likelihood scoring.

    For each problem:
    1. Build chat context via build_chat_messages (few-shot + question)
    2. Create (context, completion) chat tuples for each choice
    3. Compute log P(completion | context) via batch_score_chat
    4. Select argmax as prediction
    5. Compare to gold answer

    Uses diff-based scoring that is chat template agnostic - compares token
    sequences across choices to find answer tokens, then extracts logprobs.
    Supports both sync (Transformers) and async (vLLM) backends.

    Must be used as an async context manager:

        async with config.create_evaluator() as evaluator:
            score = await evaluator.evaluate(placement)
    """

    config: "LikelihoodEvaluatorConfig"

    def __init__(
        self, config: "LikelihoodEvaluatorConfig", *, output_dir: "Path"
    ) -> None:
        super().__init__(config, output_dir=output_dir)

    # -------------------------------------------------------------------------
    # Abstract hook implementations from TaskBasedEvaluator
    # -------------------------------------------------------------------------

    def _expected_output_type(self) -> Literal["generate_until", "multiple_choice"]:
        """Likelihood evaluators expect 'multiple_choice' tasks."""
        return "multiple_choice"

    def _build_request(
        self, task: "Task", doc: dict, idx: int, task_name: str
    ) -> tuple[ScoreChatRequest, ChatChoiceMetadata] | tuple[ScoreTextRequest, TextChoiceMetadata]:
        """Build scoring request with metadata for result processing.

        Args:
            task: The lm-eval task.
            doc: The problem document.
            idx: Problem index within the task.
            task_name: Name of the task.

        Returns:
            For chat: (ScoreChatRequest, ChatChoiceMetadata)
            For text: (ScoreTextRequest, TextChoiceMetadata)
        """
        key = f"{task_name}:{idx}"
        if self.config.use_chat_template:
            context_msgs, completion_msgs, metadata = _build_choice_chats(task, doc)
            return ScoreChatRequest(
                key=key,
                context_messages=list(context_msgs),
                completion_messages=list(completion_msgs),
            ), metadata
        else:
            context, choices, metadata = _build_choice_texts(task, doc)
            return ScoreTextRequest(
                key=key,
                context=context,
                choices=list(choices),
            ), metadata

    def _stream_to_backend_async(
        self, requests: AsyncIterable[Any]
    ) -> AsyncIterator[Any]:
        """Stream scoring requests to async backend.

        Args:
            requests: Async iterable of ScoreChatRequest or ScoreTextRequest.

        Returns:
            Async iterator of ScoreChatResult/ScoreTextResult or BackendError.
        """
        assert isinstance(self._backend, AsyncBackend)
        if self.config.use_chat_template:
            return self._backend.score_chat(requests)
        else:
            return self._backend.score_text(requests)

    def _stream_to_backend_sync(
        self, requests: Iterable[Any]
    ) -> Iterator[Any]:
        """Stream scoring requests to sync backend.

        Args:
            requests: Iterable of ScoreChatRequest or ScoreTextRequest.

        Returns:
            Iterator of ScoreChatResult/ScoreTextResult or BackendError.
        """
        assert isinstance(self._backend, SyncBackend)
        if self.config.use_chat_template:
            return self._backend.score_chat(requests)
        else:
            return self._backend.score_text(requests)

    def _process_result(
        self,
        result: ScoreChatResult | ScoreTextResult,
        context: ChoiceMetadata,
        task: "Task",
        task_name: str,
        doc: dict,
        idx: int,
    ) -> None:
        """Process scoring result by computing predictions and writing record.

        Args:
            result: ScoreChatResult or ScoreTextResult from backend.
            context: ChatChoiceMetadata or TextChoiceMetadata.
            task: The lm-eval task (unused, context has all needed data).
            task_name: Name of the task.
            doc: The problem document (unused, context has all needed data).
            idx: Problem index within the task (unused, result.key has problem_id).
        """
        assert self._trace_writer is not None

        pred = _compute_prediction(
            list(result.results), context.gold_idx, context.lengths
        )

        # Shared fields for both record types
        shared_fields = {
            "problem_id": result.key,
            "task_name": task_name,
            "raw_choices": context.raw_choices,
            "choice_token_ids": [list(r.token_ids) for r in result.results],
            "choice_logprobs": [list(r.logprobs) for r in result.results],
            "choice_answer_ranges": [r.answer_range for r in result.results],
            "choice_log_probs": pred.log_probs,
            "choice_lengths": context.lengths,
            "gold_idx": context.gold_idx,
            "pred_idx": pred.pred_idx,
            "pred_idx_norm": pred.pred_idx_norm,
            "correct": pred.correct,
            "correct_norm": pred.correct_norm,
        }

        # Create appropriate record type based on metadata
        if isinstance(context, ChatChoiceMetadata):
            record = ChatLikelihoodRecord(
                **shared_fields,
                context_messages=context.context_messages,
                choices=context.choice_messages,
            )
        else:
            record = TextLikelihoodRecord(
                **shared_fields,
                context=context.context,
                choices=context.choices,
            )

        self._trace_writer.append([record])

    # -------------------------------------------------------------------------
    # Main evaluation interface
    # -------------------------------------------------------------------------

    async def evaluate(self, parameters: LayerPlacement | None) -> float:
        """Evaluate a placement configuration using likelihood-based accuracy.

        Args:
            parameters: Layer placement to evaluate, or None for default config.
                       When None, no placement switching is performed (useful for
                       non-supernet models or backends that don't support placement).

        Returns:
            Accuracy (fraction of problems where argmax prediction is correct).
        """
        if parameters is not None:
            await self._set_placement(parameters)
        self._init_trace_writer()

        # Dispatch to base class streaming methods
        if isinstance(self._backend, AsyncBackend):
            await self._run_streaming_async()
        elif isinstance(self._backend, SyncBackend):
            await self._run_streaming_sync()
        else:
            raise TypeError(f"Unknown backend type: {type(self._backend)}")

        # Compute summary from disk (single source of truth)
        summary = summarize_likelihood(self._output_dir)
        logger.info(f"\n{summary}")

        return summary.accuracy


class LikelihoodEvaluatorConfig(TaskBasedEvaluatorConfig):
    """Config for multiple-choice tasks (MMLU, ARC, HellaSwag, etc.)."""

    _impl_class: ClassVar[type[BaseEvaluator]] = LikelihoodEvaluator

    evaluator_type: Literal["likelihood"] = "likelihood"

    # Chat template control
    use_chat_template: bool = Field(
        default=True,
        description="Whether to use chat template formatting. When False, prompts are "
        "formatted as raw text (context + choice) without any chat structure. "
        "Use False for evaluating base models or when chat template is undesired.",
    )
