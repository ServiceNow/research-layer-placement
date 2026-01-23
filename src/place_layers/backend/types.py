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

    Used by both compute_chat_log_probs and compute_text_log_probs.
    """

    token_ids: tuple[int, ...]
    """Full token sequence (context + completion)."""

    logprobs: tuple[float | None, ...]
    """Per-token logprobs (None for first token which has no prior context)."""

    answer_range: tuple[int, int]
    """(start, end) indices identifying answer tokens in token_ids."""

    total_log_prob: float
    """Sum of logprobs for answer tokens (used for prediction)."""
