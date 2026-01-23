"""Abstract base classes for generation backends.

Provides a categorical abstraction for generation backends:
- AsyncBackend: For remote servers (vLLM, sglang, OpenAI-compatible APIs)
- SyncBackend: For local models (transformers)

The command uses different execution strategies based on category:
- AsyncBackend → async worker pool with max_concurrent_requests
- SyncBackend → sync loop (sequential or batched processing)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Self, Sequence

from pydantic import BaseModel, Field

from place_layers.backend.types import ChatMessage, LogProbResult, ModelStructure

if TYPE_CHECKING:
    from place_layers.clients.types import GenerationResult


# =============================================================================
# Abstract Base Classes
# =============================================================================


class AsyncBackend(ABC):
    """Base class for async generation backends (vLLM, sglang, etc.).

    These backends communicate with remote servers via HTTP and support
    concurrent requests. Used with async worker pool execution.
    """

    @property
    @abstractmethod
    def max_concurrent_requests(self) -> int:
        """Maximum number of concurrent requests this backend supports."""
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        """Initialize backend (e.g., open HTTP session)."""
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup backend resources."""
        pass

    @abstractmethod
    async def agenerate(
        self,
        messages: list[ChatMessage],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        request_id: str = "",
    ) -> "GenerationResult":
        """Generate n completions asynchronously.

        Args:
            messages: Chat messages (list of role/content dicts).
            n: Number of completions to generate.
            max_tokens: Maximum tokens per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            request_id: Optional ID for tracking.

        Returns:
            GenerationResult with samples and prompt token IDs.
        """
        pass

    @abstractmethod
    async def compute_token_log_probs(
        self,
        traces: Sequence[tuple[Sequence[int], Sequence[int]]],
    ) -> list[float]:
        """Compute log probabilities for token-based traces.

        Args:
            traces: List of (prompt_tokens, completion_tokens) tuples.

        Returns:
            List of log probabilities (sum over completion tokens) in input order.
        """
        pass

    @abstractmethod
    async def compute_text_log_probs(
        self,
        text_groups: Sequence[tuple[str, Sequence[str]]],
    ) -> list[LogProbResult]:
        """Compute log probabilities for text continuations with diff-based scoring.

        Args:
            text_groups: List of (context, choices) tuples where:
                - context: The shared context/prompt string
                - choices: List of continuation strings to score against this context

        Returns:
            List of LogProbResult (one per choice, flattened across all groups).
        """
        pass

    @abstractmethod
    async def compute_chat_log_probs(
        self,
        chat_groups: Sequence[tuple[Sequence[ChatMessage], Sequence[ChatMessage]]],
    ) -> list[LogProbResult]:
        """Compute log probabilities for chat completions with diff-based scoring.

        Args:
            chat_groups: List of (context_messages, completion_messages) tuples where:
                - context_messages: The shared prompt (system, user, etc.)
                - completion_messages: Multiple assistant messages, one per choice

        Returns:
            List of LogProbResult (one per choice, flattened across all groups).
        """
        pass

    @abstractmethod
    async def set_placement(self, mixer_names: Sequence[str]) -> None:
        """Set the mixer placement for each layer.

        Args:
            mixer_names: Mixer name for each layer (e.g., ["attention", "mamba", ...]).

        Raises:
            NotImplementedError: If backend doesn't support dynamic placement.
        """
        pass

    @property
    @abstractmethod
    def model_structure(self) -> ModelStructure:
        """Model structure (num_layers, mixer_names).

        Raises:
            NotImplementedError: If backend doesn't support model structure discovery.
        """
        pass


class SyncBackend(ABC):
    """Base class for sync generation backends (transformers, etc.).

    These backends run locally and process requests synchronously.
    Supports both single-prompt and batched generation.
    """

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Number of prompts to batch together for generation."""
        pass

    @abstractmethod
    def __enter__(self) -> Self:
        """Initialize backend (e.g., load model)."""
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup backend resources (e.g., free GPU memory)."""
        pass

    @abstractmethod
    def batch_generate(
        self,
        messages_batch: Sequence[list[ChatMessage]],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list["GenerationResult"]:
        """Generate n completions for each message sequence in batch."""
        pass

    @abstractmethod
    def compute_token_log_probs(
        self,
        traces: Sequence[tuple[Sequence[int], Sequence[int]]],
    ) -> list[float]:
        """Compute log probabilities for token-based traces."""
        pass

    @abstractmethod
    def compute_text_log_probs(
        self,
        text_groups: Sequence[tuple[str, Sequence[str]]],
    ) -> list[LogProbResult]:
        """Compute log probabilities for text continuations with diff-based scoring."""
        pass

    @abstractmethod
    def compute_chat_log_probs(
        self,
        chat_groups: Sequence[tuple[Sequence[ChatMessage], Sequence[ChatMessage]]],
    ) -> list[LogProbResult]:
        """Compute log probabilities for chat completions with diff-based scoring.

        Args:
            chat_groups: List of (context_messages, completion_messages) tuples where:
                - context_messages: The shared prompt (system, user, etc.)
                - completion_messages: Multiple assistant messages, one per choice

        Returns:
            List of LogProbResult (one per choice, flattened across all groups).
        """
        pass

    @abstractmethod
    def set_placement(self, mixer_names: Sequence[str]) -> None:
        """Set the mixer placement for each layer."""
        pass

    @property
    @abstractmethod
    def model_structure(self) -> ModelStructure:
        """Model structure (num_layers, mixer_names)."""
        pass


# =============================================================================
# Config Base
# =============================================================================


class BaseBackendConfig(BaseModel):
    """Base configuration for generation backends."""

    model_config = {"frozen": True, "extra": "forbid"}

    backend_type: str = Field(description="Type identifier for backend")

    _backend_class: ClassVar[type[AsyncBackend] | type[SyncBackend]]

    def create_backend(self) -> AsyncBackend | SyncBackend:
        """Smart constructor: instantiate backend from config."""
        return self._backend_class(config=self)
