"""Shared types for generation clients and backends.

These data classes are used by vLLM clients and all backends
to provide a consistent interface.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, computed_field


class GeneratedSample(BaseModel):
    """A single generated completion with logprobs."""

    model_config = {"frozen": True}

    text: str = Field(description="Generated text")
    token_ids: list[int] = Field(description="Token IDs of generated text")
    token_logprobs: list[float | None] = Field(
        description="Log-probability per token (None if unavailable)"
    )
    truncated: bool = Field(description="Whether generation was truncated by max_tokens")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_logprob(self) -> float:
        """Sum of token log-probabilities."""
        return sum(lp for lp in self.token_logprobs if lp is not None)


class GenerationResult(BaseModel):
    """Result of a generation request."""

    model_config = {"frozen": True}

    key: str = Field(description="Request key for joining with input")
    samples: list[GeneratedSample] = Field(description="Generated samples")
    prompt_token_ids: list[int] = Field(description="Token IDs of the prompt")


class ScoringResult(BaseModel):
    """Result of scoring a completion (computing log probability)."""

    model_config = {"frozen": True}

    total_logprob: float = Field(description="Sum of log-probabilities for completion tokens")
    num_completion_tokens: int = Field(description="Number of completion tokens scored")
    key: str = Field(description="Request key for joining with input", default="")


class SequenceLogProbs(BaseModel):
    """Log probabilities for a full token sequence.

    Used by score_text_completion and score_chat_completion.
    Contains raw token IDs and per-token logprobs for the entire sequence
    (prompt + completion). The backend is responsible for determining
    which tokens are the "answer" via diff-based comparison.
    """

    model_config = {"frozen": True}

    token_ids: list[int] = Field(description="Full token sequence")
    logprobs: list[float | None] = Field(
        description="Per-token logprobs (None for first token which has no context)"
    )
    key: str = Field(description="Request key for joining with input", default="")
