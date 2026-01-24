"""VLLMBackend - Async backend using vLLM (local or remote)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from typing import Annotated, ClassVar, Literal, Self, Sequence

import aiohttp
from pydantic import Field

from place_layers.backend.base import (
    AsyncBackend,
    BaseBackendConfig,
    concurrent_expand_map,
    concurrent_map,
)
from place_layers.backend.types import (
    BackendError,
    ChatMessage,
    ChatScoringElement,
    GenerateRequest,
    LogProbResult,
    ModelStructure,
    ScoreChatRequest,
    ScoreChatResult,
    ScoreTextRequest,
    ScoreTextResult,
    ScoreTokensRequest,
    ScoreTokensResult,
    TextScoringElement,
)
from place_layers.clients.base_vllm import BaseVLLMClient
from place_layers.clients.local_vllm import LocalVLLMClient, LocalVLLMClientConfig
from place_layers.clients.remote_vllm import (
    RemoteVLLMClient,
    RemoteVLLMClientConfig,
    VLLMBadRequestError,
)
from place_layers.clients.types import GenerationResult, SequenceLogProbs

logger = logging.getLogger(__name__)


VLLMClientConfig = Annotated[
    LocalVLLMClientConfig | RemoteVLLMClientConfig,
    Field(discriminator="client_type"),
]


class VLLMBackend(AsyncBackend):
    """Async generation backend using vLLM (local in-process or remote HTTP).

    Supports both:
    - Local mode: In-process vLLM engine with OpenAI serving layer
    - Remote mode: HTTP client for remote vLLM server

    Local mode supports Apriel2 models with runtime mixer placement switching.
    """

    def __init__(self, config: "VLLMBackendConfig") -> None:
        self._config = config
        self._client: BaseVLLMClient

        if isinstance(config.client, LocalVLLMClientConfig):
            self._client = LocalVLLMClient(config.client)
        else:
            self._client = RemoteVLLMClient(config.client)

        self._current_placement: tuple[str, ...] | None = None

    @property
    def max_concurrent_requests(self) -> int:
        """Maximum number of concurrent requests."""
        return self._config.client.max_concurrent_requests

    async def __aenter__(self) -> Self:
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._client.__aexit__(exc_type, exc_val, exc_tb)

    @property
    def model_structure(self) -> ModelStructure:
        """Model structure (num_layers, mixer_names).

        Only available in local mode with Apriel2 supernet models.
        """
        if not isinstance(self._client, LocalVLLMClient):
            raise NotImplementedError(
                "Model structure discovery is not supported for remote vLLM backend."
            )
        if not self._client.supports_placement:
            raise NotImplementedError(
                "Model structure discovery requires an Apriel2 supernet model."
            )
        return ModelStructure(
            num_layers=self._client.num_layers or 0,
            mixer_names=self._client.mixer_names or (),
        )

    def generate(
        self,
        requests: Iterable[GenerateRequest] | AsyncIterable[GenerateRequest | None],
    ) -> AsyncIterator[GenerationResult | BackendError]:
        """Generate completions, yielding results as they complete.

        Results may arrive in any order. Use request_id to join with input.
        Each request contains its own generation params (n, max_tokens, etc.).
        Concurrency is managed internally via semaphore.

        On transient errors (bad request, timeout, server error), yields
        BackendError instead of killing the stream.
        """

        async def process_one(req: GenerateRequest) -> GenerationResult | BackendError:
            try:
                return await self._client.chat_generate(
                    messages=req.messages,
                    n=req.n,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    request_id=req.key,
                )
            except VLLMBadRequestError as e:
                logger.warning(f"Bad request for {req.key}: {e}")
                return BackendError(key=req.key, error_type="bad_request", message=str(e))
            except asyncio.TimeoutError as e:
                logger.warning(f"Timeout for {req.key}: {e}")
                return BackendError(key=req.key, error_type="timeout", message=str(e))
            except aiohttp.ClientError as e:
                logger.warning(f"Server error for {req.key}: {e}")
                return BackendError(key=req.key, error_type="server_error", message=str(e))

        return concurrent_map(requests, process_one, self.max_concurrent_requests)

    async def score_tokens(
        self,
        requests: Iterable[ScoreTokensRequest]
        | AsyncIterable[ScoreTokensRequest | None],
    ) -> AsyncIterator[ScoreTokensResult | BackendError]:
        """Score token sequences, yielding results as they complete.

        Results may arrive in any order. Concurrency is managed internally
        via semaphore based on max_concurrent_requests.

        On transient errors, yields BackendError instead of killing the stream.
        """

        async def process_one(req: ScoreTokensRequest) -> ScoreTokensResult | BackendError:
            try:
                result = await self._client.score_token_completion(
                    req.prompt_tokens, req.completion_tokens
                )
                return ScoreTokensResult(key=req.key, log_prob=result.total_logprob)
            except VLLMBadRequestError as e:
                logger.warning(f"Bad request for {req.key}: {e}")
                return BackendError(key=req.key, error_type="bad_request", message=str(e))
            except asyncio.TimeoutError as e:
                logger.warning(f"Timeout for {req.key}: {e}")
                return BackendError(key=req.key, error_type="timeout", message=str(e))
            except aiohttp.ClientError as e:
                logger.warning(f"Server error for {req.key}: {e}")
                return BackendError(key=req.key, error_type="server_error", message=str(e))

        async for result in concurrent_map(
            requests, process_one, self.max_concurrent_requests
        ):
            yield result

    def _build_diff_results(
        self, results: Sequence[SequenceLogProbs]
    ) -> list[LogProbResult]:
        """Build LogProbResults using diff-based scoring.

        Finds where token sequences diverge across choices and computes
        total log probability for the differing (answer) tokens.
        """
        from place_layers.core.token_utils import find_differing_ranges

        token_sequences = [list(r.token_ids) for r in results]
        answer_ranges = find_differing_ranges(token_sequences)

        log_prob_results: list[LogProbResult] = []
        for r, (start, end) in zip(results, answer_ranges):
            answer_logprobs = [
                lp
                for pos in range(start, end)
                if pos < len(r.logprobs) and (lp := r.logprobs[pos]) is not None
            ]
            log_prob_results.append(
                LogProbResult(
                    token_ids=r.token_ids,
                    logprobs=r.logprobs,
                    answer_range=(start, end),
                    total_log_prob=sum(answer_logprobs),
                )
            )
        return log_prob_results

    async def score_text(
        self,
        requests: Iterable[ScoreTextRequest] | AsyncIterable[ScoreTextRequest | None],
    ) -> AsyncIterator[ScoreTextResult | BackendError]:
        """Score text continuations (multiple-choice), yielding as complete.

        Results may arrive in any order. Each result contains all choices
        for that request (diff-based scoring requires all choices together).

        On transient errors, yields BackendError instead of killing the stream.
        If any choice fails, the entire request fails (diff-based scoring
        requires all choices).
        """

        def expand(
            req: ScoreTextRequest,
        ) -> tuple[str, list[TextScoringElement]]:
            # Fan-out: one request -> list of text scoring elements
            return (
                req.key,
                [
                    TextScoringElement(context=req.context, continuation=choice)
                    for choice in req.choices
                ],
            )

        async def process_one(
            element: TextScoringElement,
        ) -> SequenceLogProbs | BackendError:
            try:
                return await self._client.score_text_completion(
                    element.context, element.continuation
                )
            except VLLMBadRequestError as e:
                # Return error with empty key - combine will fill in the real key
                return BackendError(key="", error_type="bad_request", message=str(e))
            except asyncio.TimeoutError as e:
                return BackendError(key="", error_type="timeout", message=str(e))
            except aiohttp.ClientError as e:
                return BackendError(key="", error_type="server_error", message=str(e))

        def combine(
            key: str, results: list[SequenceLogProbs | BackendError]
        ) -> ScoreTextResult | BackendError:
            # Check for any errors - if any choice failed, fail the whole request
            for r in results:
                if isinstance(r, BackendError):
                    logger.warning(f"Error scoring {key}: {r.message}")
                    return BackendError(
                        key=key, error_type=r.error_type, message=r.message
                    )

            # All succeeded - build diff results
            success_results = [r for r in results if isinstance(r, SequenceLogProbs)]
            if not success_results:
                return ScoreTextResult(key=key, results=[])
            return ScoreTextResult(key=key, results=self._build_diff_results(success_results))

        async for result in concurrent_expand_map(
            requests, expand, process_one, combine, self.max_concurrent_requests
        ):
            yield result

    async def score_chat(
        self,
        requests: Iterable[ScoreChatRequest] | AsyncIterable[ScoreChatRequest | None],
    ) -> AsyncIterator[ScoreChatResult | BackendError]:
        """Score chat completions (multiple-choice), yielding as complete.

        Results may arrive in any order. Each result contains all choices
        for that request (diff-based scoring requires all choices together).

        On transient errors, yields BackendError instead of killing the stream.
        If any choice fails, the entire request fails (diff-based scoring
        requires all choices).
        """

        def expand(
            req: ScoreChatRequest,
        ) -> tuple[str, list[ChatScoringElement]]:
            # Fan-out: one request -> list of chat scoring elements
            return (
                req.key,
                [
                    ChatScoringElement(
                        context_messages=list(req.context_messages),
                        completion_message=comp_msg,
                    )
                    for comp_msg in req.completion_messages
                ],
            )

        async def process_one(
            element: ChatScoringElement,
        ) -> SequenceLogProbs | BackendError:
            try:
                return await self._client.score_chat_completion(
                    element.context_messages, [element.completion_message]
                )
            except VLLMBadRequestError as e:
                return BackendError(key="", error_type="bad_request", message=str(e))
            except asyncio.TimeoutError as e:
                return BackendError(key="", error_type="timeout", message=str(e))
            except aiohttp.ClientError as e:
                return BackendError(key="", error_type="server_error", message=str(e))

        def combine(
            key: str, results: list[SequenceLogProbs | BackendError]
        ) -> ScoreChatResult | BackendError:
            # Check for any errors - if any choice failed, fail the whole request
            for r in results:
                if isinstance(r, BackendError):
                    logger.warning(f"Error scoring {key}: {r.message}")
                    return BackendError(
                        key=key, error_type=r.error_type, message=r.message
                    )

            # All succeeded - build diff results
            success_results = [r for r in results if isinstance(r, SequenceLogProbs)]
            if not success_results:
                return ScoreChatResult(key=key, results=[])
            return ScoreChatResult(key=key, results=self._build_diff_results(success_results))

        async for result in concurrent_expand_map(
            requests, expand, process_one, combine, self.max_concurrent_requests
        ):
            yield result

    async def set_placement(self, mixer_names: Sequence[str]) -> None:
        """Set mixer placement for Apriel2 supernet model.

        Only available in local mode with Apriel2 supernet models.
        """
        if not isinstance(self._client, LocalVLLMClient):
            raise NotImplementedError(
                "Dynamic mixer placement is not supported for remote vLLM backend."
            )

        placement_tuple = tuple(mixer_names)
        if self._current_placement == placement_tuple:
            return

        await self._client.set_placement(list(mixer_names))
        self._current_placement = placement_tuple


class VLLMBackendConfig(BaseBackendConfig):
    """Configuration for vLLM generation backend (local or remote)."""

    _backend_class: ClassVar[type[AsyncBackend]] = VLLMBackend  # type: ignore[assignment]

    backend_type: Literal["vllm"] = "vllm"
    client: VLLMClientConfig = Field(
        description="Client configuration (LocalVLLMClientConfig or RemoteVLLMClientConfig)"
    )
