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
    GenerateChatRequest as BackendGenerateChatRequest,
    GenerateTextRequest as BackendGenerateTextRequest,
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


class GenerateChatRequest(BackendGenerateChatRequest):
    """Request for chat generation (extends backend type with placement)."""

    placement: list[str] | None = Field(
        default=None,
        description="Mixer placement for each layer.",
    )


class GenerateTextRequest(BackendGenerateTextRequest):
    """Request for text generation (extends backend type with placement)."""

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
