"""Abstract base class for vLLM clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from place_layers.core.types import ChatMessage
    from place_layers.clients.types import GenerationResult, ScoringResult, SequenceLogProbs


class BaseVLLMClient(ABC):
    """Abstract base for vLLM clients (local and remote).

    Defines the common interface for generation and scoring operations.
    Must be used as an async context manager.
    """

    @abstractmethod
    async def __aenter__(self) -> Self:
        """Initialize the client."""
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up the client."""
        ...

    @abstractmethod
    async def chat_generate(
        self,
        messages: list[ChatMessage],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        request_id: str = "",
    ) -> GenerationResult:
        """Generate completions from chat messages.

        Args:
            messages: Chat messages.
            n: Number of completions to generate.
            max_tokens: Maximum tokens per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            request_id: Optional ID for tracking.

        Returns:
            GenerationResult with samples and prompt token IDs.
        """
        ...

    @abstractmethod
    async def score_token_completion(
        self,
        prompt_tokens: Sequence[int],
        completion_tokens: Sequence[int],
        request_id: str = "",
    ) -> ScoringResult:
        """Score a completion given prompt tokens.

        Args:
            prompt_tokens: Token IDs for the prompt/context.
            completion_tokens: Token IDs for the completion to score.
            request_id: Optional identifier for tracking.

        Returns:
            ScoringResult with total log probability.
        """
        ...

    @abstractmethod
    async def score_text_completion(
        self,
        context: str,
        continuation: str,
        request_id: str = "",
    ) -> SequenceLogProbs:
        """Score a text continuation.

        Returns token IDs and logprobs for the full sequence (context + continuation).
        The backend is responsible for determining which tokens are the "answer"
        via diff-based comparison across multiple choices.

        Args:
            context: The context/prompt text.
            continuation: The continuation text to score.
            request_id: Optional identifier for tracking.

        Returns:
            SequenceLogProbs with token IDs and per-token logprobs.
        """
        ...

    @abstractmethod
    async def score_chat_completion(
        self,
        context_messages: list[ChatMessage],
        completion_messages: list[ChatMessage],
        request_id: str = "",
    ) -> SequenceLogProbs:
        """Score chat completion messages.

        Returns token IDs and logprobs for the full sequence (context + completion).
        The backend is responsible for determining which tokens are the "answer"
        via diff-based comparison across multiple choices.

        Args:
            context_messages: Context messages (system, user, etc.).
            completion_messages: Completion messages to score (typically assistant).
            request_id: Optional identifier for tracking.

        Returns:
            SequenceLogProbs with token IDs and per-token logprobs.
        """
        ...
