"""Async client for remote vLLM server via HTTP.

Uses vLLM's OpenAI-compatible API endpoints.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal, Self

import aiohttp
from pydantic import BaseModel, Field

from place_layers.core.types import ChatMessage
from place_layers.clients.base_vllm import BaseVLLMClient
from place_layers.clients.types import (
    GeneratedSample,
    GenerationResult,
    ScoringResult,
    SequenceLogProbs,
)

logger = logging.getLogger(__name__)


class VLLMBadRequestError(Exception):
    """Raised when vLLM returns HTTP 400 (e.g., prompt too long)."""

    pass


class RemoteVLLMClientConfig(BaseModel):
    """Configuration for remote vLLM client."""

    model_config = {"frozen": True}

    client_type: Literal["remote"] = Field(default="remote", description="Client type discriminator")
    base_url: str = Field(description="vLLM server base URL (e.g., 'http://localhost:8000')")
    model: str = Field(description="Model name as registered in vLLM")
    api_key: str = Field(default="EMPTY", description="API key (use 'EMPTY' if not required)")
    timeout_seconds: int = Field(default=300, description="Request timeout")
    max_concurrent_requests: int = Field(default=32, description="Max concurrent requests")


class RemoteVLLMClient(BaseVLLMClient):
    """Async client for remote vLLM server via HTTP.

    Communicates with vLLM's OpenAI-compatible API. The server handles
    chat templates, tokenization, and model inference.

    Must be used as an async context manager:

        async with RemoteVLLMClient(config) as client:
            result = await client.chat_generate(...)
    """

    def __init__(self, config: RemoteVLLMClientConfig) -> None:
        self.config = config
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> Self:
        """Initialize the HTTP session."""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the HTTP session."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Get the session, raising if not in context manager."""
        if self._session is None:
            raise RuntimeError(
                "RemoteVLLMClient must be used as an async context manager: "
                "async with RemoteVLLMClient(config) as client: ..."
            )
        return self._session

    # =========================================================================
    # Generation
    # =========================================================================

    async def chat_generate(
        self,
        messages: list[ChatMessage],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        key: str,
    ) -> GenerationResult:
        """Generate completions from chat messages.

        Uses /v1/chat/completions - the server applies the chat template.

        Args:
            messages: Chat messages.
            n: Number of completions to generate.
            max_tokens: Maximum tokens per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            key: Request key for joining with input.

        Returns:
            GenerationResult with samples and prompt token IDs.
        """
        session = self._ensure_session()
        url = f"{self.config.base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [m.model_dump() for m in messages],
            "n": n,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "logprobs": True,
            "return_token_ids": True,
        }

        async with session.post(url, json=payload) as resp:
            if resp.status == 400:
                error_text = await resp.text()
                raise VLLMBadRequestError(error_text)
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(
                    f"vLLM chat request failed with status {resp.status}: {error_text}"
                )
            data = await resp.json()

        prompt_token_ids: list[int] = data.get("prompt_token_ids") or []

        samples: list[GeneratedSample] = []
        for choice in data["choices"]:
            text: str = choice["message"]["content"]
            token_ids: list[int] = choice.get("token_ids") or []
            truncated = choice.get("finish_reason") == "length"

            logprobs_data = choice.get("logprobs") or {}
            content_logprobs = logprobs_data.get("content") or []
            token_logprobs = [item["logprob"] for item in content_logprobs]

            samples.append(
                GeneratedSample(
                    text=text,
                    token_ids=token_ids,
                    token_logprobs=token_logprobs,
                    truncated=truncated,
                )
            )

        return GenerationResult(
            key=key,
            samples=samples,
            prompt_token_ids=prompt_token_ids,
        )

    async def text_generate(
        self,
        prompt: str,
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        key: str,
        ignore_eos: bool = False,
    ) -> GenerationResult:
        """Generate completions from a text prompt.

        Uses /v1/completions endpoint.

        Args:
            prompt: Text prompt.
            n: Number of completions to generate.
            max_tokens: Maximum tokens per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            key: Request key for joining with input.

        Returns:
            GenerationResult with samples and prompt token IDs.
        """
        session = self._ensure_session()
        url = f"{self.config.base_url.rstrip('/')}/v1/completions"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "n": n,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "logprobs": 1,
            "ignore_eos": ignore_eos,
        }

        async with session.post(url, json=payload) as resp:
            if resp.status == 400:
                error_text = await resp.text()
                raise VLLMBadRequestError(error_text)
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(
                    f"vLLM completion request failed with status {resp.status}: {error_text}"
                )
            data = await resp.json()

        samples: list[GeneratedSample] = []
        for choice in data["choices"]:
            text: str = choice["text"]
            token_ids: list[int] = choice.get("token_ids") or []
            truncated = choice.get("finish_reason") == "length"

            logprobs_data = choice.get("logprobs") or {}
            token_logprobs_raw = logprobs_data.get("token_logprobs") or []
            token_logprobs = [lp for lp in token_logprobs_raw if lp is not None]

            samples.append(
                GeneratedSample(
                    text=text,
                    token_ids=token_ids,
                    token_logprobs=token_logprobs,
                    truncated=truncated,
                )
            )

        # Get prompt token IDs from first choice if available
        prompt_token_ids: list[int] = []
        if data["choices"]:
            prompt_token_ids = data["choices"][0].get("prompt_token_ids") or []

        return GenerationResult(
            key=key,
            samples=samples,
            prompt_token_ids=prompt_token_ids,
        )

    # =========================================================================
    # Scoring
    # =========================================================================

    async def score_token_completion(
        self,
        prompt_tokens: Sequence[int],
        completion_tokens: Sequence[int],
        key: str,
    ) -> ScoringResult:
        """Score a completion given prompt tokens.

        Args:
            prompt_tokens: Token IDs for the prompt/context.
            completion_tokens: Token IDs for the completion to score.
            key: Request key for joining with input.

        Returns:
            ScoringResult with total log probability.
        """
        session = self._ensure_session()
        url = f"{self.config.base_url.rstrip('/')}/v1/completions"

        full_tokens = list(prompt_tokens) + list(completion_tokens)
        payload = {
            "model": self.config.model,
            "prompt": full_tokens,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 1,
        }

        async with session.post(url, json=payload) as resp:
            if resp.status == 400:
                error_text = await resp.text()
                raise VLLMBadRequestError(error_text)
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(
                    f"vLLM score request failed: {resp.status}: {error_text}"
                )
            data = await resp.json()

        choice = data["choices"][0]
        logprobs_data = choice.get("logprobs", {})
        token_logprobs = logprobs_data.get("token_logprobs", [])

        prompt_len = len(prompt_tokens)
        completion_len = len(completion_tokens)
        completion_logprobs = token_logprobs[prompt_len : prompt_len + completion_len]
        valid_logprobs = [lp for lp in completion_logprobs if lp is not None]

        return ScoringResult(
            total_logprob=sum(valid_logprobs),
            num_completion_tokens=len(valid_logprobs),
            key=key,
        )

    async def score_text_completion(
        self,
        context: str,
        continuation: str,
        key: str,
    ) -> SequenceLogProbs:
        """Score a text continuation.

        Returns token IDs and logprobs for the full sequence (context + continuation).
        The backend is responsible for determining which tokens are the "answer"
        via diff-based comparison across multiple choices.

        Args:
            context: The context/prompt text.
            continuation: The continuation text to score.
            key: Request key for joining with input.

        Returns:
            SequenceLogProbs with token IDs and per-token logprobs.
        """
        session = self._ensure_session()
        url = f"{self.config.base_url.rstrip('/')}/v1/completions"

        full_text = context + continuation

        payload = {
            "model": self.config.model,
            "prompt": full_text,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 1,
        }

        async with session.post(url, json=payload) as resp:
            if resp.status == 400:
                error_text = await resp.text()
                raise VLLMBadRequestError(error_text)
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(
                    f"vLLM score request failed: {resp.status}: {error_text}"
                )
            data = await resp.json()

        choice = data["choices"][0]
        logprobs_data = choice.get("logprobs", {})
        token_logprobs = logprobs_data.get("token_logprobs", [])
        tokens = logprobs_data.get("tokens", [])

        # Extract token IDs from prompt_logprobs
        prompt_logprobs = choice.get("prompt_logprobs", [])
        token_ids: list[int] = []
        for i, token_str in enumerate(tokens):
            if i < len(prompt_logprobs) and prompt_logprobs[i] is not None:
                for token_id_str, info in prompt_logprobs[i].items():
                    if info.get("decoded_token") == token_str:
                        token_ids.append(int(token_id_str))
                        break
                else:
                    token_ids.append(-1)
            else:
                token_ids.append(-1)

        return SequenceLogProbs(
            token_ids=token_ids,
            logprobs=token_logprobs,
            key=key,
        )

    async def score_chat_completion(
        self,
        context_messages: list[ChatMessage],
        completion_messages: list[ChatMessage],
        key: str,
    ) -> SequenceLogProbs:
        """Score chat completion messages.

        Returns token IDs and logprobs for the full sequence (context + completion).
        The backend is responsible for determining which tokens are the "answer"
        via diff-based comparison across multiple choices.

        Args:
            context_messages: Context messages (system, user, etc.).
            completion_messages: Completion messages to score (typically assistant).
            key: Request key for joining with input.

        Returns:
            SequenceLogProbs with token IDs and per-token logprobs.
        """
        session = self._ensure_session()
        url = f"{self.config.base_url.rstrip('/')}/v1/chat/completions"

        full_messages = context_messages + completion_messages

        # prompt_logprobs gives us logprobs for all input tokens
        # max_tokens=1 is the minimum vLLM accepts; generated token is discarded
        payload = {
            "model": self.config.model,
            "messages": [m.model_dump() for m in full_messages],
            "max_tokens": 1,
            "prompt_logprobs": 1,
            "temperature": 0,
        }

        async with session.post(url, json=payload) as resp:
            if resp.status == 400:
                error_text = await resp.text()
                raise VLLMBadRequestError(error_text)
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(
                    f"vLLM chat score request failed: {resp.status}: {error_text}"
                )
            data = await resp.json()

        # Extract token IDs and logprobs from prompt_logprobs
        prompt_logprobs = data.get("prompt_logprobs", [])

        token_ids: list[int] = []
        logprobs: list[float | None] = []

        for lp in prompt_logprobs:
            if lp is None:
                token_ids.append(-1)
                logprobs.append(None)
            else:
                token_id = int(list(lp.keys())[0])
                token_logprob = lp[list(lp.keys())[0]]["logprob"]
                token_ids.append(token_id)
                logprobs.append(token_logprob)

        return SequenceLogProbs(
            token_ids=token_ids,
            logprobs=logprobs,
            key=key,
        )

