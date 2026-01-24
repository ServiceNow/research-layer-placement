"""API request/response schemas for backend REST server.

These schemas extend the backend request/result types, adding optional
placement fields for requests. Single-request endpoints enable
server-side batching via the adapter.

Scoring requests subclass backend types and add placement.
Scoring responses are just aliases to backend result types.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from place_layers.backend.types import (
    ChatMessage,
    GenerateRequest as BackendGenerateRequest,
    LogProbResult,
    ScoreChatRequest as BackendScoreChatRequest,
    ScoreChatResult,
    ScoreTextRequest as BackendScoreTextRequest,
    ScoreTextResult,
    ScoreTokensRequest as BackendScoreTokensRequest,
    ScoreTokensResult,
)

# =============================================================================
# Shared Types - Re-export from backend
# =============================================================================

# Use backend ChatMessage directly (identical to old ChatMessageSchema)
ChatMessageSchema = ChatMessage

# =============================================================================
# Generation
# =============================================================================


class GenerateRequest(BackendGenerateRequest):
    """Request for generation (extends backend type with params + placement)."""

    # Generation params (batch-level in backend, but per-request over REST)
    n: int = Field(default=1, description="Number of completions to generate", ge=1)
    max_tokens: int = Field(
        default=512, description="Maximum tokens per completion", ge=1
    )
    temperature: float = Field(
        default=0.0, description="Sampling temperature", ge=0.0, le=2.0
    )
    top_p: float = Field(
        default=1.0, description="Nucleus sampling parameter", ge=0.0, le=1.0
    )

    # Optional placement
    placement: list[str] | None = Field(
        default=None,
        description="Mixer placement for each layer.",
    )


# GenerationResult is returned directly from backend (model_dump())


# =============================================================================
# Token Scoring
# =============================================================================


class ScoreTokensRequest(BackendScoreTokensRequest):
    """Request for token scoring (extends backend type with placement)."""

    placement: list[str] | None = Field(
        default=None,
        description="Mixer placement for each layer.",
    )


# Response is identical to backend result type
ScoreTokensResponse = ScoreTokensResult


# =============================================================================
# Text Scoring
# =============================================================================


class ScoreTextRequest(BackendScoreTextRequest):
    """Request for text scoring (extends backend type with placement)."""

    placement: list[str] | None = Field(
        default=None,
        description="Mixer placement for each layer.",
    )


# Re-export LogProbResult for use in response types
LogProbResultSchema = LogProbResult

# Response is identical to backend result type
ScoreTextResponse = ScoreTextResult


# =============================================================================
# Chat Scoring
# =============================================================================


class ScoreChatRequest(BackendScoreChatRequest):
    """Request for chat scoring (extends backend type with placement)."""

    placement: list[str] | None = Field(
        default=None,
        description="Mixer placement for each layer.",
    )


# Response is identical to backend result type
ScoreChatResponse = ScoreChatResult


# =============================================================================
# Server Info
# =============================================================================


class InfoResponse(BaseModel):
    """Server info response."""

    model_config = {"frozen": True}

    model_path: str = Field(description="Path to the loaded model")
    batch_size: int = Field(description="Configured batch size for generation")
    supports_placement: bool = Field(
        description="Whether this server supports dynamic mixer placement"
    )
    current_placement: list[str] | None = Field(
        description="Current mixer placement, or None if not set"
    )
    num_layers: int | None = Field(
        default=None,
        description="Number of layers with switchable mixers (None if not a supernet)",
    )
    mixer_names: list[str] | None = Field(
        default=None,
        description="Available mixer types (None if not a supernet)",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    model_config = {"frozen": True}

    status: str = Field(description="Server status")
