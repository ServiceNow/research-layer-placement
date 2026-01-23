"""VLLMBackend - Async backend using vLLM (local or remote)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Self, Sequence

from pydantic import Field

from place_layers.backend.base import AsyncBackend, BaseBackendConfig, concurrent_map
from place_layers.backend.types import (
    ChatMessage,
    ChatScoringRequest,
    ChatScoringResult,
    LogProbResult,
    ModelStructure,
    TextScoringRequest,
    TextScoringResult,
    TokenScoringRequest,
    TokenScoringResult,
)
from place_layers.clients.local_vllm import LocalVLLMClient, LocalVLLMClientConfig
from place_layers.clients.remote_vllm import RemoteVLLMClient, RemoteVLLMClientConfig

if TYPE_CHECKING:
    from place_layers.clients.types import GenerationResult, SequenceLogProbs


VLLMClientConfig = Annotated[
    LocalVLLMClientConfig | RemoteVLLMClientConfig,
    Field(discriminator="client_type"),
]


class VLLMBackend:
    """Async generation backend using vLLM (local in-process or remote HTTP).

    Supports both:
    - Local mode: In-process vLLM engine with OpenAI serving layer
    - Remote mode: HTTP client for remote vLLM server

    Local mode supports Apriel2 models with runtime mixer placement switching.
    """

    def __init__(self, config: "VLLMBackendConfig") -> None:
        self._config = config
        self._client: LocalVLLMClient | RemoteVLLMClient

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
        """Generate completions for a conversation."""
        return await self._client.chat_generate(
            messages=messages,  # type: ignore[arg-type]
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            request_id=request_id,
        )

    async def score_tokens(
        self,
        requests: Iterable[TokenScoringRequest],
    ) -> AsyncIterator[TokenScoringResult]:
        """Score token sequences, yielding results as they complete.

        Results may arrive in any order. Concurrency is managed internally
        via semaphore based on max_concurrent_requests.
        """

        async def process_one(req: TokenScoringRequest) -> TokenScoringResult:
            result = await self._client.score_token_completion(
                list(req.prompt_tokens), list(req.completion_tokens)
            )
            return TokenScoringResult(key=req.key, log_prob=result.total_logprob)

        async for result in concurrent_map(
            requests, process_one, self.max_concurrent_requests
        ):
            yield result

    def _build_diff_results(
        self, results: Sequence["SequenceLogProbs"]
    ) -> tuple[LogProbResult, ...]:
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
        return tuple(log_prob_results)

    async def score_text(
        self,
        requests: Iterable[TextScoringRequest],
    ) -> AsyncIterator[TextScoringResult]:
        """Score text continuations (multiple-choice), yielding as complete.

        Results may arrive in any order. Each result contains all choices
        for that request (diff-based scoring requires all choices together).
        """

        async def process_one(req: TextScoringRequest) -> TextScoringResult:
            if not req.choices:
                return TextScoringResult(key=req.key, results=())

            tasks = [
                self._client.score_text_completion(req.context, choice)
                for choice in req.choices
            ]
            results = await asyncio.gather(*tasks)
            return TextScoringResult(
                key=req.key, results=self._build_diff_results(results)
            )

        async for result in concurrent_map(
            requests, process_one, self.max_concurrent_requests
        ):
            yield result

    async def score_chat(
        self,
        requests: Iterable[ChatScoringRequest],
    ) -> AsyncIterator[ChatScoringResult]:
        """Score chat completions (multiple-choice), yielding as complete.

        Results may arrive in any order. Each result contains all choices
        for that request (diff-based scoring requires all choices together).
        """

        async def process_one(req: ChatScoringRequest) -> ChatScoringResult:
            if not req.completion_messages:
                return ChatScoringResult(key=req.key, results=())

            tasks = [
                self._client.score_chat_completion(
                    list(req.context_messages),  # type: ignore[arg-type]
                    [comp_msg],  # Single message wrapped in list
                )
                for comp_msg in req.completion_messages
            ]
            results = await asyncio.gather(*tasks)
            return ChatScoringResult(
                key=req.key, results=self._build_diff_results(results)
            )

        async for result in concurrent_map(
            requests, process_one, self.max_concurrent_requests
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
