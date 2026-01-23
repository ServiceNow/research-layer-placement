"""VLLMBackend - Async backend using vLLM (local or remote)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Self, Sequence

from pydantic import Field

from place_layers.backend.base import AsyncBackend, BaseBackendConfig
from place_layers.backend.types import ChatMessage, LogProbResult, ModelStructure
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

    async def compute_token_log_probs(
        self, traces: Sequence[tuple[Sequence[int], Sequence[int]]]
    ) -> list[float]:
        """Compute log probabilities for token-based traces."""
        if not traces:
            return []

        tasks = [
            self._client.score_token_completion(list(prompt), list(completion))
            for prompt, completion in traces
        ]
        results = await asyncio.gather(*tasks)
        return [r.total_logprob for r in results]

    def _build_diff_results(
        self, results: Sequence["SequenceLogProbs"]
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

    async def compute_text_log_probs(
        self, text_groups: Sequence[tuple[str, Sequence[str]]]
    ) -> list[LogProbResult]:
        """Compute log probabilities for text continuations with diff-based scoring."""
        if not text_groups:
            return []

        all_results: list[LogProbResult] = []

        for context, choices in text_groups:
            if not choices:
                continue

            tasks = [
                self._client.score_text_completion(context, choice)
                for choice in choices
            ]
            results = await asyncio.gather(*tasks)
            all_results.extend(self._build_diff_results(results))

        return all_results

    async def compute_chat_log_probs(
        self,
        chat_groups: Sequence[tuple[Sequence[ChatMessage], Sequence[ChatMessage]]],
    ) -> list[LogProbResult]:
        """Compute log probabilities for chat completions with diff-based scoring.

        Args:
            chat_groups: Sequence of (context_messages, completion_messages) tuples.
                context_messages: The prompt (system, user, etc.)
                completion_messages: Multiple assistant messages, one per choice.
        """
        if not chat_groups:
            return []

        all_results: list[LogProbResult] = []

        for context_msgs, completion_msgs in chat_groups:
            if not completion_msgs:
                continue

            tasks = [
                self._client.score_chat_completion(
                    list(context_msgs),  # type: ignore[arg-type]
                    [comp_msg],  # Single message wrapped in list
                )
                for comp_msg in completion_msgs
            ]
            results = await asyncio.gather(*tasks)
            all_results.extend(self._build_diff_results(results))

        return all_results

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
