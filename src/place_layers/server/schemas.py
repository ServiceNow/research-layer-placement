"""API request/response schemas for TransformersBackend REST server.

These schemas define the wire format for communication between the REST client
and server. They are designed to be compatible with the existing backend
interface while supporting the placement-in-request pattern.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessageSchema(BaseModel):
    """A single message in a chat conversation."""

    model_config = {"frozen": True}

    role: Literal["system", "user", "assistant"] = Field(
        description="Role of the message sender"
    )
    content: str = Field(description="Message content")


class GenerateRequest(BaseModel):
    """Request for single-prompt generation."""

    model_config = {"frozen": True}

    messages: list[ChatMessageSchema] = Field(
        description="Chat messages (conversation history)"
    )
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
    placement: list[str] | None = Field(
        default=None,
        description="Mixer placement for each layer (e.g., ['attention', 'mamba', ...]). "
        "If provided, server will set this placement before generation.",
    )
    request_id: str = Field(default="", description="Optional request identifier")


class BatchGenerateRequest(BaseModel):
    """Request for batch generation (multiple prompts)."""

    model_config = {"frozen": True}

    messages_batch: list[list[ChatMessageSchema]] = Field(
        description="List of conversation histories, one per prompt"
    )
    n: int = Field(
        default=1, description="Number of completions per prompt", ge=1
    )
    max_tokens: int = Field(
        default=512, description="Maximum tokens per completion", ge=1
    )
    temperature: float = Field(
        default=0.0, description="Sampling temperature", ge=0.0, le=2.0
    )
    top_p: float = Field(
        default=1.0, description="Nucleus sampling parameter", ge=0.0, le=1.0
    )
    placement: list[str] | None = Field(
        default=None,
        description="Mixer placement for each layer. "
        "If provided, server will set this placement before generation.",
    )


class TraceSchema(BaseModel):
    """A single trace for log probability computation."""

    model_config = {"frozen": True}

    prompt_tokens: list[int] = Field(description="Token IDs of the prompt")
    completion_tokens: list[int] = Field(description="Token IDs of the completion")


class ComputeLogProbsRequest(BaseModel):
    """Request for computing log probabilities of traces."""

    model_config = {"frozen": True}

    traces: list[TraceSchema] = Field(
        description="List of (prompt_tokens, completion_tokens) traces to score"
    )
    placement: list[str] | None = Field(
        default=None,
        description="Mixer placement for each layer. "
        "If provided, server will set this placement before scoring.",
    )


class ComputeLogProbsResponse(BaseModel):
    """Response from log probability computation."""

    model_config = {"frozen": True}

    log_probs: list[float] = Field(
        description="Log probabilities for each trace (sum over completion tokens)"
    )


class SetPlacementRequest(BaseModel):
    """Request to set mixer placement (for debugging/testing)."""

    model_config = {"frozen": True}

    mixer_names: list[str] = Field(
        description="Mixer name for each layer (e.g., ['attention', 'mamba', ...])"
    )


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


class ChatLogProbResultSchema(BaseModel):
    """Detailed result from chat log-prob computation."""

    model_config = {"frozen": True}

    token_ids: list[int] = Field(
        description="Full token sequence for this chat (context + completion)"
    )
    logprobs: list[float | None] = Field(
        description="Per-token logprobs (None for first token which has no prior context)"
    )
    answer_range: tuple[int, int] = Field(
        description="(start, end) indices identifying answer tokens in token_ids"
    )
    total_log_prob: float = Field(
        description="Total log probability (sum of answer token logprobs)"
    )


class ChatPairSchema(BaseModel):
    """A single chat pair for log-prob computation."""

    model_config = {"frozen": True}

    context_messages: list[ChatMessageSchema] = Field(
        description="Context messages (e.g., question and instruction)"
    )
    completion_messages: list[ChatMessageSchema] = Field(
        description="Completion messages (e.g., the answer choice)"
    )


class ComputeChatLogProbsRequest(BaseModel):
    """Request for computing log probabilities of chat completions."""

    model_config = {"frozen": True}

    chats: list[ChatPairSchema] = Field(
        description="List of (context, completion) chat pairs to score"
    )
    placement: list[str] | None = Field(
        default=None,
        description="Mixer placement for each layer. "
        "If provided, server will set this placement before scoring.",
    )


class ComputeChatLogProbsResponse(BaseModel):
    """Response from chat log-prob computation."""

    model_config = {"frozen": True}

    results: list[ChatLogProbResultSchema] = Field(
        description="Detailed log-prob results for each chat pair"
    )


class ErrorResponse(BaseModel):
    """Error response."""

    model_config = {"frozen": True}

    error: str = Field(description="Error message")
    detail: str | None = Field(default=None, description="Additional error details")
