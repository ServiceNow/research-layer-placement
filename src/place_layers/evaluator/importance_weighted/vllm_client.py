"""Async client for vLLM OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Self

import aiohttp
from pydantic import BaseModel, Field, computed_field

logger = logging.getLogger(__name__)


class VLLMBadRequestError(Exception):
    """Raised when vLLM returns HTTP 400 (e.g., prompt too long)."""

    pass


class VLLMClientConfig(BaseModel):
    """Configuration for vLLM client."""

    model_config = {"frozen": True}

    base_url: str = Field(description="vLLM server base URL (e.g., 'http://localhost:8000')")
    model: str = Field(description="Model name as registered in vLLM")
    api_key: str = Field(description="API key (use 'EMPTY' if not required)", default="EMPTY")
    timeout_seconds: int = Field(description="Request timeout", default=300)
    max_concurrent_requests: int = Field(description="Max concurrent requests to vLLM", default=32)


class GeneratedSample(BaseModel):
    """A single generated completion with logprobs."""

    model_config = {"frozen": True}

    text: str = Field(description="Generated text")
    token_ids: list[int] = Field(description="Token IDs of generated text")
    token_logprobs: list[float] = Field(description="Log-probability per token")
    truncated: bool = Field(description="Whether generation was truncated by max_tokens")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_logprob(self) -> float:
        """Sum of token log-probabilities."""
        return sum(self.token_logprobs)


class GenerationResult(BaseModel):
    """Result of a generation request."""

    model_config = {"frozen": True}

    request_id: str = Field(description="Request identifier")
    samples: list[GeneratedSample] = Field(description="Generated samples")
    prompt_token_ids: list[int] = Field(description="Token IDs of the prompt")


class ScoringResult(BaseModel):
    """Result of scoring a completion (computing log probability)."""

    model_config = {"frozen": True}

    total_logprob: float = Field(description="Sum of log-probabilities for completion tokens")
    num_completion_tokens: int = Field(description="Number of completion tokens scored")
    request_id: str = Field(description="Request identifier", default="")


class VLLMClient:
    """Async client for vLLM's OpenAI-compatible chat completions API.

    Uses /v1/chat/completions which applies the model's chat template.

    Must be used as an async context manager:

        async with VLLMClient(config) as client:
            result = await client.chat_generate(...)
    """

    def __init__(self, config: VLLMClientConfig) -> None:
        self.config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
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
                "VLLMClient must be used as an async context manager: "
                "async with VLLMClient(config) as client: ..."
            )
        return self._session

    async def chat_generate(
        self,
        messages: list[dict[str, str]],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        request_id: str = "",
    ) -> GenerationResult:
        """Generate n completions using chat completions API.

        Uses /v1/chat/completions which applies the model's chat template.

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
        session = self._ensure_session()
        url = f"{self.config.base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "n": n,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "logprobs": True,
            "return_token_ids": True,
        }

        async with self._semaphore:
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

        # prompt_token_ids is at the response root level for chat completions
        prompt_token_ids: list[int] = data.get("prompt_token_ids") or []

        samples: list[GeneratedSample] = []
        for choice in data["choices"]:
            text: str = choice["message"]["content"]
            token_ids: list[int] = choice.get("token_ids") or []
            truncated = choice.get("finish_reason") == "length"

            # Extract logprobs from the logprobs.content array
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
            request_id=request_id,
            samples=samples,
            prompt_token_ids=prompt_token_ids,
        )

    async def tokenize(self, text: str) -> Sequence[int]:
        """Tokenize text using vLLM's tokenize endpoint.

        Args:
            text: Text to tokenize.

        Returns:
            Sequence of token IDs.
        """
        session = self._ensure_session()
        url = f"{self.config.base_url.rstrip('/')}/tokenize"
        payload = {
            "model": self.config.model,
            "prompt": text,
        }

        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"Tokenize failed: {error_text}")
            data = await resp.json()

        return data.get("tokens", [])

    async def score_completion(
        self,
        prompt_tokens: Sequence[int],
        completion_tokens: Sequence[int],
        request_id: str = "",
    ) -> ScoringResult:
        """Compute log probability of completion given prompt.

        Uses vLLM's echo mode: sends the full sequence (prompt + completion)
        and requests logprobs for all tokens, then extracts just the
        completion token logprobs.

        Args:
            prompt_tokens: Token IDs for the prompt/context.
            completion_tokens: Token IDs for the completion to score.
            request_id: Optional identifier for tracking.

        Returns:
            ScoringResult with total log probability and token count.
        """
        session = self._ensure_session()
        url = f"{self.config.base_url.rstrip('/')}/v1/completions"

        # Concatenate prompt and completion tokens
        full_tokens = list(prompt_tokens) + list(completion_tokens)

        payload = {
            "model": self.config.model,
            "prompt": full_tokens,  # vLLM accepts token IDs directly
            "max_tokens": 0,  # Don't generate new tokens
            "echo": True,  # Return logprobs for prompt tokens
            "logprobs": 1,  # Request logprobs
        }

        async with self._semaphore:
            async with session.post(url, json=payload) as resp:
                if resp.status == 400:
                    error_text = await resp.text()
                    raise VLLMBadRequestError(error_text)
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(
                        f"vLLM score request failed with status {resp.status}: {error_text}"
                    )
                data = await resp.json()

        # Extract logprobs from response
        # With echo=True, we get logprobs for all tokens including prompt
        choice = data["choices"][0]
        logprobs_data = choice.get("logprobs", {})
        token_logprobs = logprobs_data.get("token_logprobs", [])

        # The first token has no logprob (it's the starting token)
        # Completion logprobs start at position len(prompt_tokens)
        prompt_len = len(prompt_tokens)
        completion_len = len(completion_tokens)

        # Extract logprobs for completion tokens only
        # token_logprobs[i] is logprob of token[i] given tokens[:i]
        completion_logprobs = token_logprobs[prompt_len : prompt_len + completion_len]

        # Filter out None values (first token might be None)
        valid_logprobs = [lp for lp in completion_logprobs if lp is not None]
        total_logprob = sum(valid_logprobs)

        return ScoringResult(
            total_logprob=total_logprob,
            num_completion_tokens=len(valid_logprobs),
            request_id=request_id,
        )

