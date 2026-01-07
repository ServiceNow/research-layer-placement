"""Shared utilities for lm-eval task handling.

This module provides common functionality used by both trace precomputation
and generative evaluation:

- Task specification and loading
- Document retrieval from tasks
- Chat message construction (with consistent few-shot formatting)
- Correctness checking via lm-eval's process_results

The key design goal is ensuring consistent prompt construction across
trace generation and evaluation, so that prompts match exactly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from pydantic import BaseModel, Field

from place_layers.evaluator.importance_weighted.generation_backend import ChatMessage

if TYPE_CHECKING:
    from lm_eval.api.task import Task

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols and Types
# =============================================================================


class Docs(Protocol):
    """Protocol for lm-eval document collections.

    HuggingFace Dataset and list[dict] both satisfy this protocol.
    We only require __len__ and __getitem__ - the minimal interface
    needed for indexed access and resumption logic.
    """

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> dict: ...


# =============================================================================
# Task Configuration
# =============================================================================


class TaskSpec(BaseModel):
    """Configuration for a single lm-eval task.

    Allows per-task overrides of few-shot count, description, and sampling strategy.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    task: str = Field(description="lm-eval task name (e.g., 'gsm8k')")
    num_fewshot: int | None = Field(
        default=None, description="Override number of few-shot examples"
    )
    description: str | None = Field(
        default=None, description="Override system/task description"
    )
    fewshot_sampler: Literal["first_n", "random", "balanced"] | None = Field(
        default=None, description="Few-shot sampling strategy"
    )

    def to_lm_eval_config(self, default_num_fewshot: int) -> dict:
        """Convert to lm-eval task config dict."""
        config: dict = {"task": self.task}
        config["num_fewshot"] = (
            self.num_fewshot if self.num_fewshot is not None else default_num_fewshot
        )
        if self.description is not None:
            config["description"] = self.description
        if self.fewshot_sampler is not None:
            config["fewshot_config"] = {"sampler": self.fewshot_sampler}
        return config


# Union type for task specification (string or TaskSpec object)
TaskSpecInput = str | TaskSpec


# =============================================================================
# Task Helper Functions
# =============================================================================


def normalize_task_spec(task: TaskSpecInput, default_num_fewshot: int) -> dict:
    """Convert a single task spec to lm-eval config dict.

    Args:
        task: Either a task name string or a TaskSpec with overrides.
        default_num_fewshot: Default few-shot count if not specified.

    Returns:
        Dict suitable for lm-eval's get_task_dict().
    """
    if isinstance(task, str):
        return {"task": task, "num_fewshot": default_num_fewshot}
    return task.to_lm_eval_config(default_num_fewshot)


def get_task_name(task_spec: TaskSpecInput) -> str:
    """Extract task name from task spec.

    Args:
        task_spec: Either a task name string or a TaskSpec.

    Returns:
        The task name string.
    """
    if isinstance(task_spec, str):
        return task_spec
    return task_spec.task


def get_docs(task: "Task", split: str) -> Docs:
    """Get documents from task for the specified split.

    Tries the requested split first, then falls back to whatever is available.
    Returns an indexable sequence (HuggingFace Dataset or list of dicts).

    Args:
        task: The lm-eval task object.
        split: Preferred split ('test' or 'validation').

    Returns:
        Indexable document collection.

    Raises:
        RuntimeError: If task has no test or validation docs.
        TypeError: If docs don't support indexing (required by lm-eval).
    """
    docs: Any
    if split == "test" and task.has_test_docs():
        docs = task.test_docs()
    elif split == "validation" and task.has_validation_docs():
        docs = task.validation_docs()
    elif task.has_test_docs():
        docs = task.test_docs()
    elif task.has_validation_docs():
        docs = task.validation_docs()
    else:
        raise RuntimeError(f"Task '{task.config.task}' has no test or validation docs")

    # Verify docs supports indexing (required by resumption logic and lm-eval)
    try:
        _ = len(docs)
        if len(docs) > 0:
            _ = docs[0]
    except (TypeError, KeyError) as e:
        raise TypeError(
            f"Task '{task.config.task}' docs must support len() and __getitem__ "
            f"(got {type(docs).__name__}). This is required by lm-eval and our "
            f"resumption logic."
        ) from e

    return cast(Docs, docs)


# =============================================================================
# Prompt Construction
# =============================================================================


def build_chat_messages(
    task: "Task",
    doc: dict,
    num_fewshot: int,
    fewshot_as_multiturn: bool = True,
) -> list[ChatMessage]:
    """Build chat messages for a problem with few-shot examples.

    This is the single source of truth for prompt construction. Both trace
    precomputation and evaluation MUST use this function to ensure prompts
    are identical.

    Args:
        task: The lm-eval task.
        doc: The problem document.
        num_fewshot: Number of few-shot examples.
        fewshot_as_multiturn: If True, few-shot examples appear as alternating
            user/assistant turns. If False, all examples are concatenated in
            a single user message. Default True matches typical chat model usage.

    Returns:
        List of ChatMessage dicts with role/content keys.
    """
    # Get few-shot context using lm-eval's sampler
    # fewshot_as_multiturn=True: each example is a separate user/assistant turn
    # fewshot_as_multiturn=False: all examples concatenated in one user message
    messages: list[ChatMessage] = task.sampler.get_chat_context(
        doc, num_fewshot=num_fewshot, fewshot_as_multiturn=fewshot_as_multiturn
    )

    # Append the actual question as the final user message
    question = task.doc_to_text(doc)
    messages.append({"role": "user", "content": question})

    return messages


# =============================================================================
# Answer Extraction
# =============================================================================


def extract_answer_region(
    completion: str,
    pattern: str | None = None,
) -> str:
    """Extract the answer region from a completion using a regex pattern.

    This allows models with structured output formats (e.g., Apriel's
    [BEGIN FINAL RESPONSE]...<|end|>) to have their answers correctly
    parsed by lm-eval.

    Note: For models with special end tokens (like Apriel's <|end|>), this
    extraction IS necessary. lm-eval's regex `(?s).*#### ` leaves content
    after the answer (e.g., "18\\n<|end|>"), causing exact_match to fail.
    The extraction pattern excludes the end token, leaving just the answer.

    Args:
        completion: The full completion text.
        pattern: Regex pattern with a capture group for the answer region.
                 If None, returns completion unchanged.
                 Example: r"\\[BEGIN FINAL RESPONSE\\](.*?)(?:<\\|end\\|>|$)"

    Returns:
        The extracted answer region (stripped of leading/trailing whitespace),
        or full completion if no pattern/no match.
    """
    if pattern is None:
        # Still strip whitespace for consistency with exact_match comparison
        return completion.strip()

    import re

    try:
        match = re.search(pattern, completion, re.DOTALL)
        if match:
            # Return first capture group if present, else full match
            extracted = match.group(1) if match.lastindex else match.group(0)
            # IMPORTANT: strip() is critical for exact_match comparison.
            # Without it, trailing newlines (e.g., "18\n" vs "18") cause
            # false negatives because lm-eval's regexes don't strip whitespace.
            extracted = extracted.strip()
            logger.debug(f"Extracted answer region: {extracted[:100]}...")
            return extracted
        else:
            logger.debug(f"Pattern '{pattern}' did not match, using full completion")
            return completion.strip()
    except re.error as e:
        logger.warning(f"Invalid regex pattern '{pattern}': {e}. Using full completion.")
        return completion.strip()


# =============================================================================
# Correctness Checking
# =============================================================================


def check_correctness(
    task: "Task",
    doc: dict,
    completion: str,
    answer_extraction_pattern: str | None = None,
) -> bool | None:
    """Check if completion is correct using task's process_results.

    For generate_until tasks, lm-eval returns metrics like:
    - GSM8K: {"exact_match": 0 or 1}
    - MATH: {"exact_match": 0 or 1, "exact_match_original": 0 or 1}
    - AIME: {"exact_match": 0 or 1}

    We prefer "exact_match" when present (uses symbolic math verification
    for MATH tasks), falling back to the first binary metric.

    Args:
        task: The lm-eval task.
        doc: The problem document.
        completion: The generated completion text.
        answer_extraction_pattern: Optional regex pattern to extract the answer
            region before passing to lm-eval. Use for models with structured
            output formats (e.g., Apriel's [BEGIN FINAL RESPONSE]...<|end|>).

    Returns:
        True if correct, False if incorrect, None if correctness cannot
        be determined from the task's metrics.
    """
    # Extract answer region if pattern provided
    completion = extract_answer_region(completion, answer_extraction_pattern)

    metrics: dict[str, Any] | None = task.process_results(doc, [completion])

    if metrics is None:
        logger.warning(
            "process_results returned None. Record will be saved with correct=None."
        )
        return None

    # Prefer exact_match - it's the standard metric and uses symbolic
    # verification for math tasks (more accurate than string matching)
    if "exact_match" in metrics:
        value = metrics["exact_match"]
        if isinstance(value, (int, float)) and value in (0, 1, 0.0, 1.0):
            return value == 1 or value == 1.0

    # Fallback: first binary metric we find
    for value in metrics.values():
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1, 0.0, 1.0):
            return value == 1 or value == 1.0

    logger.warning(
        f"Could not determine correctness from metrics {metrics}. "
        f"Record will be saved with correct=None."
    )
    return None


# =============================================================================
# Task Loading
# =============================================================================


def load_task(
    task_spec: TaskSpecInput,
    default_num_fewshot: int,
    seed: int | None = None,
) -> "Task":
    """Load an lm-eval task from a task spec.

    Args:
        task_spec: Task name or TaskSpec with overrides.
        default_num_fewshot: Default few-shot count.
        seed: Random seed for few-shot sampling (optional).

    Returns:
        Configured lm-eval Task object.
    """
    from lm_eval.tasks import TaskManager, get_task_dict

    task_name = get_task_name(task_spec)
    task_config = normalize_task_spec(task_spec, default_num_fewshot)
    task_manager = TaskManager()
    task_dict = get_task_dict([task_config], task_manager=task_manager)
    task = task_dict[task_name]

    # Set seed if provided
    if seed is not None:
        task.set_fewshot_seed(seed)

    return task
