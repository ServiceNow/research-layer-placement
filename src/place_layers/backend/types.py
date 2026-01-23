"""Backend types and data classes.

Shared types used by all backend implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict


class ChatMessage(TypedDict):
    """A single message in a chat conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


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
class LogProbResult:
    """Detailed result from log-prob computation.

    Contains all token-level information needed for observability:
    - Full token sequence (after formatting/chat template applied)
    - Per-token logprobs
    - Answer range (which tokens were identified as "the answer")
    - Total score (sum of answer logprobs)

    Used by score_text and score_chat.
    """

    token_ids: tuple[int, ...]
    """Full token sequence (context + completion)."""

    logprobs: tuple[float | None, ...]
    """Per-token logprobs (None for first token which has no prior context)."""

    answer_range: tuple[int, int]
    """(start, end) indices identifying answer tokens in token_ids."""

    total_log_prob: float
    """Sum of logprobs for answer tokens (used for prediction)."""


# =============================================================================
# Scoring Request/Result Types
#
# These types carry identity (key) with the data, enabling streaming without
# order preservation. Backend yields results in any order (first-completed),
# caller joins on key.
# =============================================================================


@dataclass(frozen=True)
class TokenScoringRequest:
    """Request to score a single token sequence (prompt + completion)."""

    key: str
    """Unique identifier for this request (returned with result)."""

    prompt_tokens: tuple[int, ...]
    """Token IDs for the prompt/context."""

    completion_tokens: tuple[int, ...]
    """Token IDs for the completion to score."""


@dataclass(frozen=True)
class TokenScoringResult:
    """Result of scoring a token sequence."""

    key: str
    """Identifier matching the request."""

    log_prob: float
    """Total log probability of completion tokens."""


@dataclass(frozen=True)
class TextScoringRequest:
    """Request to score text continuations (multiple-choice)."""

    key: str
    """Unique identifier for this request (returned with result)."""

    context: str
    """Shared context/prompt string."""

    choices: tuple[str, ...]
    """Continuation strings to score (one per choice)."""


@dataclass(frozen=True)
class TextScoringResult:
    """Result of scoring text continuations."""

    key: str
    """Identifier matching the request."""

    results: tuple[LogProbResult, ...]
    """Scoring results, one per choice in request order."""


@dataclass(frozen=True)
class ChatScoringRequest:
    """Request to score chat completions (multiple-choice)."""

    key: str
    """Unique identifier for this request (returned with result)."""

    context_messages: tuple[ChatMessage, ...]
    """Shared context messages (system, user, etc.)."""

    completion_messages: tuple[ChatMessage, ...]
    """Assistant messages to score (one per choice)."""


@dataclass(frozen=True)
class ChatScoringResult:
    """Result of scoring chat completions."""

    key: str
    """Identifier matching the request."""

    results: tuple[LogProbResult, ...]
    """Scoring results, one per choice in request order."""
