"""Backend types and data classes.

Shared types used by all backend implementations.

Request/result types use Pydantic BaseModel for serialization compatibility.
Server schemas can subclass these to add placement fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from place_layers.core.types import ChatMessage


class BackendError(BaseModel):
    """Error from a backend operation.

    Used as a union member with result types to enable graceful error handling
    in streaming APIs. When a request fails (e.g., prompt too long, server error),
    the backend yields a BackendError instead of killing the stream.

    Callers can check `isinstance(result, BackendError)` and handle accordingly
    (e.g., log as SkippedProblem).
    """

    model_config = {"frozen": True}

    key: str = Field(description="Request key for joining with input")
    error_type: Literal["bad_request", "timeout", "server_error"] = Field(
        description="Category of error"
    )
    message: str = Field(description="Human-readable error message")


class GenerateRequest(BaseModel):
    """Request to generate completions for a chat conversation."""

    model_config = {"frozen": True}

    key: str = Field(description="Unique identifier for this request")
    messages: list[ChatMessage] = Field(description="Chat messages")

    # Generation parameters (requests with same params can be batched together)
    n: int = Field(default=1, description="Number of completions to generate")
    max_tokens: int = Field(default=256, description="Maximum tokens to generate")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, description="Top-p (nucleus) sampling")


@dataclass(frozen=True)
class ModelStructure:
    """Model structure for placement optimization.

    Generic representation of model structure that backends expose.
    Decoupled from Apriel2-specific details.
    """

    num_layers: int
    """Number of layers with switchable mixers."""

    mixer_names: tuple[str, ...]
    """Available mixer types (e.g., ('attention', 'gdn', 'kda'))."""


@dataclass(frozen=True)
class SequenceScoringRequest:
    """Request to score a single token sequence with specified answer range.

    This is the internal/low-level scoring request used by _score_sequences.
    Unlike TokenScoringRequest (which has separate prompt/completion),
    this has pre-concatenated tokens with an explicit answer range
    (determined by diff-based scoring for multiple-choice).
    """

    tokens: tuple[int, ...]
    """Full token sequence to score."""

    answer_range: tuple[int, int]
    """(start, end) indices for tokens that contribute to total_log_prob."""


class TextScoringElement(BaseModel):
    """Element for scoring a text continuation.

    Used internally by VLLMBackend for the expand/process_one pattern.
    Text-level (pre-tokenization) - vLLM handles tokenization server-side.
    """

    model_config = {"frozen": True}

    context: str = Field(description="Context/prompt text")
    continuation: str = Field(description="Continuation text to score")


class ChatScoringElement(BaseModel):
    """Element for scoring a chat completion.

    Used internally by VLLMBackend for the expand/process_one pattern.
    Message-level (pre-tokenization) - vLLM handles tokenization server-side.
    """

    model_config = {"frozen": True}

    context_messages: list[ChatMessage] = Field(description="Context messages")
    completion_message: ChatMessage = Field(description="Completion message to score")


class LogProbResult(BaseModel):
    """Detailed result from log-prob computation.

    Contains all token-level information needed for observability:
    - Full token sequence (after formatting/chat template applied)
    - Per-token logprobs
    - Answer range (which tokens were identified as "the answer")
    - Total score (sum of answer logprobs)

    Used by score_text and score_chat.
    """

    model_config = {"frozen": True}

    token_ids: list[int] = Field(description="Full token sequence")
    logprobs: list[float | None] = Field(
        description="Per-token logprobs (None for first token)"
    )
    answer_range: tuple[int, int] = Field(
        description="(start, end) indices for answer tokens"
    )
    total_log_prob: float = Field(description="Sum of logprobs for answer tokens")


# =============================================================================
# Scoring Request/Result Types
#
# These types carry identity (key) with the data, enabling streaming without
# order preservation. Backend yields results in any order (first-completed),
# caller joins on key.
# =============================================================================


class ScoreTokensRequest(BaseModel):
    """Request to score a single token sequence (prompt + completion)."""

    model_config = {"frozen": True}

    key: str = Field(description="Unique identifier for this request")
    prompt_tokens: list[int] = Field(description="Token IDs for the prompt")
    completion_tokens: list[int] = Field(description="Token IDs to score")


class ScoreTokensResult(BaseModel):
    """Result of scoring a token sequence."""

    model_config = {"frozen": True}

    key: str = Field(description="Identifier matching the request")
    log_prob: float = Field(description="Total log probability of completion tokens")


class ScoreTextRequest(BaseModel):
    """Request to score text continuations (multiple-choice)."""

    model_config = {"frozen": True}

    key: str = Field(description="Unique identifier for this request")
    context: str = Field(description="Shared context/prompt string")
    choices: list[str] = Field(description="Continuation strings to score")


class ScoreTextResult(BaseModel):
    """Result of scoring text continuations."""

    model_config = {"frozen": True}

    key: str = Field(description="Identifier matching the request")
    results: list[LogProbResult] = Field(
        description="Scoring results, one per choice"
    )


class ScoreChatRequest(BaseModel):
    """Request to score chat completions (multiple-choice)."""

    model_config = {"frozen": True}

    key: str = Field(description="Unique identifier for this request")
    context_messages: list[ChatMessage] = Field(
        description="Shared context messages"
    )
    completion_messages: list[ChatMessage] = Field(
        description="Assistant messages to score"
    )


class ScoreChatResult(BaseModel):
    """Result of scoring chat completions."""

    model_config = {"frozen": True}

    key: str = Field(description="Identifier matching the request")
    results: list[LogProbResult] = Field(
        description="Scoring results, one per choice"
    )
