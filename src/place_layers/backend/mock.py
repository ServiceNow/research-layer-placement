"""Mock backend for testing.

Provides a replay-based backend that doesn't require a real model or GPU.
Responses are configured via Hydra YAML and replayed in order.

Example config:
    backend_type: mock
    num_layers: 24
    replay:
      generate_chat:
        - samples:
            - text: "The answer is 42"
              token_ids: [1, 2, 3]
              token_logprobs: [-0.1, -0.2, -0.1]
              truncated: false
          prompt_token_ids: [100, 101]
      score_chat:
        - results:
            - token_ids: [1, 2]
              logprobs: [null, -0.5]
              answer_range: [1, 2]
              total_log_prob: -0.5
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Sequence
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, Field

from place_layers.backend.base import AsyncBackend, BaseBackendConfig
from place_layers.backend.types import (
    BackendError,
    GenerateChatRequest,
    GenerateTextRequest,
    ModelStructure,
    ScoreChatRequest,
    ScoreChatResult,
    ScoreTextRequest,
    ScoreTextResult,
    ScoreTokensRequest,
    ScoreTokensResult,
)
from place_layers.clients.types import GenerationResult

logger = logging.getLogger(__name__)


class ReplayState:
    """Tracks replay position for each endpoint."""

    def __init__(self, replay: "ReplayConfig") -> None:
        self._replay = replay
        self._indices: dict[str, int] = {}

    def next_response(self, endpoint: str) -> Any | None:
        """Get next response for endpoint, or None if exhausted."""
        responses = getattr(self._replay, endpoint, None)
        if not responses:
            return None

        idx = self._indices.get(endpoint, 0)
        if idx >= len(responses):
            return None  # Exhausted

        self._indices[endpoint] = idx + 1
        return responses[idx]

    def reset(self, endpoint: str | None = None) -> None:
        """Reset replay position for endpoint (or all if None)."""
        if endpoint is None:
            self._indices.clear()
        else:
            self._indices.pop(endpoint, None)


class ReplayConfig(BaseModel):
    """Replay data for mock backend endpoints.

    Each field is a list of response models that will be yielded in order.
    Models are serialized to/from YAML via Pydantic.

    All fields must be provided via Hydra YAML - no defaults in code.
    """

    model_config = {"extra": "forbid"}

    generate_chat: list[GenerationResult | BackendError] = Field(
        description="Responses to replay for generate_chat",
    )
    generate_text: list[GenerationResult | BackendError] = Field(
        description="Responses to replay for generate_text",
    )
    score_tokens: list[ScoreTokensResult | BackendError] = Field(
        description="Responses to replay for score_tokens",
    )
    score_text: list[ScoreTextResult | BackendError] = Field(
        description="Responses to replay for score_text",
    )
    score_chat: list[ScoreChatResult | BackendError] = Field(
        description="Responses to replay for score_chat",
    )


class MockBackend(AsyncBackend):
    """Mock backend that replays configured responses.

    Responses come from config.replay and are yielded in order.
    Raises RuntimeError when replay is exhausted.

    Useful for testing evaluator pipeline logic without GPU.
    """

    def __init__(self, config: "MockBackendConfig") -> None:
        self._config = config
        self._replay = ReplayState(config.replay)
        self._current_placement: list[str] | None = None
        if config.num_layers and config.mixer_names:
            self._current_placement = [config.mixer_names[0]] * config.num_layers

    @property
    def max_concurrent_requests(self) -> int:
        """Maximum concurrent requests (configurable)."""
        return self._config.max_concurrent_requests

    async def __aenter__(self) -> Self:
        """Reset replay state on entry."""
        self._replay.reset()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """No-op cleanup."""
        pass

    async def generate_chat(
        self,
        requests: Iterable[GenerateChatRequest]
        | AsyncIterable[GenerateChatRequest | None],
    ) -> AsyncIterator[GenerationResult | BackendError]:
        """Replay configured chat completions."""
        async for request in self._iter_requests(requests):
            if request is None:
                continue

            response = self._replay.next_response("generate_chat")
            if response is None:
                raise RuntimeError(
                    f"Mock backend: generate_chat replay exhausted at request {request.key}"
                )

            if isinstance(response, BackendError):
                yield response
            else:
                yield GenerationResult(
                    request_id=request.key,
                    samples=response.samples,
                    prompt_token_ids=response.prompt_token_ids,
                )

    async def generate_text(
        self,
        requests: Iterable[GenerateTextRequest]
        | AsyncIterable[GenerateTextRequest | None],
    ) -> AsyncIterator[GenerationResult | BackendError]:
        """Replay configured text completions."""
        async for request in self._iter_requests(requests):
            if request is None:
                continue

            response = self._replay.next_response("generate_text")
            if response is None:
                raise RuntimeError(
                    f"Mock backend: generate_text replay exhausted at request {request.key}"
                )

            if isinstance(response, BackendError):
                yield response
            else:
                yield GenerationResult(
                    request_id=request.key,
                    samples=response.samples,
                    prompt_token_ids=response.prompt_token_ids,
                )

    async def score_tokens(
        self,
        requests: Iterable[ScoreTokensRequest]
        | AsyncIterable[ScoreTokensRequest | None],
    ) -> AsyncIterator[ScoreTokensResult | BackendError]:
        """Replay configured token scores."""
        async for request in self._iter_requests(requests):
            if request is None:
                continue

            response = self._replay.next_response("score_tokens")
            if response is None:
                raise RuntimeError(
                    f"Mock backend: score_tokens replay exhausted at request {request.key}"
                )

            if isinstance(response, BackendError):
                yield response
            else:
                yield ScoreTokensResult(
                    key=request.key,
                    log_prob=response.log_prob,
                )

    async def score_text(
        self,
        requests: Iterable[ScoreTextRequest] | AsyncIterable[ScoreTextRequest | None],
    ) -> AsyncIterator[ScoreTextResult | BackendError]:
        """Replay configured text scores."""
        async for request in self._iter_requests(requests):
            if request is None:
                continue

            response = self._replay.next_response("score_text")
            if response is None:
                raise RuntimeError(
                    f"Mock backend: score_text replay exhausted at request {request.key}"
                )

            if isinstance(response, BackendError):
                yield response
            else:
                yield ScoreTextResult(
                    key=request.key,
                    results=response.results,
                )

    async def score_chat(
        self,
        requests: Iterable[ScoreChatRequest] | AsyncIterable[ScoreChatRequest | None],
    ) -> AsyncIterator[ScoreChatResult | BackendError]:
        """Replay configured chat scores."""
        async for request in self._iter_requests(requests):
            if request is None:
                continue

            response = self._replay.next_response("score_chat")
            if response is None:
                raise RuntimeError(
                    f"Mock backend: score_chat replay exhausted at request {request.key}"
                )

            if isinstance(response, BackendError):
                yield response
            else:
                yield ScoreChatResult(
                    key=request.key,
                    results=response.results,
                )

    async def set_placement(self, placement: Sequence[str]) -> None:
        """Set the mock placement."""
        if self._config.num_layers is None:
            raise NotImplementedError("Mock backend has no layers configured")
        if len(placement) != self._config.num_layers:
            raise ValueError(
                f"Placement length {len(placement)} doesn't match "
                f"num_layers {self._config.num_layers}"
            )
        self._current_placement = list(placement)

    @property
    def placement(self) -> list[str] | None:
        """Current mock placement."""
        return self._current_placement

    @property
    def model_structure(self) -> ModelStructure:
        """Return configured model structure."""
        if self._config.num_layers is None or self._config.mixer_names is None:
            raise NotImplementedError("Mock backend has no model structure configured")
        return ModelStructure(
            num_layers=self._config.num_layers,
            mixer_names=self._config.mixer_names,
        )

    async def _iter_requests[T](
        self, requests: Iterable[T] | AsyncIterable[T]
    ) -> AsyncIterator[T]:
        """Iterate over requests, handling both sync and async iterables."""
        if isinstance(requests, AsyncIterable):
            async for request in requests:
                yield request
        else:
            for request in requests:
                yield request


class MockBackendConfig(BaseBackendConfig):
    """Configuration for replay-based mock backend.

    Responses are configured via the `replay` field and yielded in order.
    Raises RuntimeError when replay is exhausted.

    All fields must be provided via Hydra YAML - no defaults in code.
    """

    _backend_class: ClassVar[type[MockBackend]] = MockBackend

    backend_type: Literal["mock"] = "mock"

    num_layers: int | None = Field(
        description="Number of layers to simulate. None disables model_structure.",
    )
    mixer_names: tuple[str, ...] | None = Field(
        description="Available mixer names. None disables model_structure.",
    )
    max_concurrent_requests: int = Field(
        description="Maximum concurrent requests to simulate.",
    )

    replay: ReplayConfig = Field(
        description="Replay data for each endpoint. Responses are yielded in order.",
    )
