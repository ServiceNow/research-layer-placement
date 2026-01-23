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


class TextScoringResult(BaseModel):
    """Detailed result of text-based log probability computation."""

    model_config = {"frozen": True}

    token_ids: tuple[int, ...] = Field(description="Full token sequence (context + continuation)")
    logprobs: tuple[float | None, ...] = Field(
        description="Per-token logprobs (None for first token)"
    )
    answer_range: tuple[int, int] = Field(
        description="(start, end) indices for continuation tokens"
    )
    total_logprob: float = Field(description="Sum of log-probabilities for continuation tokens")
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

    async def text_generate(
        self,
        prompt: str,
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        request_id: str = "",
    ) -> GenerationResult:
        """Generate n completions from text prompt without chat template.

        Uses /v1/completions which treats the prompt as raw text.

        Args:
            prompt: Raw text prompt string.
            n: Number of completions to generate.
            max_tokens: Maximum tokens per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            request_id: Optional ID for tracking.

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
            "logprobs": 1,  # Request logprobs (vLLM returns top logprob by default)
            "echo": False,  # Don't echo the prompt
        }

        async with self._semaphore:
            async with session.post(url, json=payload) as resp:
                if resp.status == 400:
                    error_text = await resp.text()
                    raise VLLMBadRequestError(error_text)
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(
                        f"vLLM completions request failed with status {resp.status}: {error_text}"
                    )
                data = await resp.json()

        # Extract prompt_token_ids from usage (completions API)
        usage = data.get("usage") or {}
        prompt_tokens_count = usage.get("prompt_tokens", 0)

        # For completions API, we need to tokenize the prompt to get token IDs
        prompt_token_ids = await self.tokenize(prompt)

        samples: list[GeneratedSample] = []
        for choice in data["choices"]:
            text: str = choice["text"]
            truncated = choice.get("finish_reason") == "length"

            # Extract token IDs and logprobs from the choice
            # vLLM completions API may return these differently than chat
            logprobs_data = choice.get("logprobs") or {}
            token_logprobs = logprobs_data.get("token_logprobs") or []
            tokens = logprobs_data.get("tokens") or []

            # Tokenize the completion text to get token IDs
            # (vLLM completions might not return token_ids directly)
            if tokens:
                # If we have string tokens, we need to convert them
                # For now, tokenize the completion text
                token_ids = await self.tokenize(text)
            else:
                token_ids = []

            samples.append(
                GeneratedSample(
                    text=text,
                    token_ids=list(token_ids),
                    token_logprobs=token_logprobs,
                    truncated=truncated,
                )
            )

        return GenerationResult(
            request_id=request_id,
            samples=samples,
            prompt_token_ids=list(prompt_token_ids),
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

    async def score_text_completion(
        self,
        context: str,
        continuation: str,
        request_id: str = "",
    ) -> TextScoringResult:
        """Compute log probability of text continuation with full token detail.

        Unlike score_completion() which requires pre-tokenized inputs, this method
        accepts raw text and uses vLLM's text_offset in the response to determine
        where the continuation starts.

        Args:
            context: The context/prompt text.
            continuation: The continuation text to score.
            request_id: Optional identifier for tracking.

        Returns:
            TextScoringResult with token IDs, per-token logprobs, answer range,
            and total log probability.
        """
        session = self._ensure_session()
        url = f"{self.config.base_url.rstrip('/')}/v1/completions"

        # Send full text - vLLM will tokenize internally
        full_text = context + continuation
        continuation_start_char = len(context)

        payload = {
            "model": self.config.model,
            "prompt": full_text,
            "max_tokens": 0,  # Don't generate new tokens
            "echo": True,  # Return logprobs for prompt tokens
            "logprobs": 1,  # Request logprobs (includes text_offset)
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

        # Extract logprobs data
        choice = data["choices"][0]
        logprobs_data = choice.get("logprobs", {})
        token_logprobs = logprobs_data.get("token_logprobs", [])
        text_offset = logprobs_data.get("text_offset", [])

        # Extract tokens list
        tokens = logprobs_data.get("tokens", [])

        # Extract token IDs from prompt_logprobs
        # vLLM's prompt_logprobs has token IDs as dictionary keys
        # We find the actual token by matching decoded_token to tokens[i]
        prompt_logprobs = choice.get("prompt_logprobs", [])
        prompt_token_ids: list[int] = []
        for i, token_str in enumerate(tokens):
            if i < len(prompt_logprobs) and prompt_logprobs[i] is not None:
                # Find the entry where decoded_token matches our token
                for token_id_str, info in prompt_logprobs[i].items():
                    if info.get("decoded_token") == token_str:
                        prompt_token_ids.append(int(token_id_str))
                        break
                else:
                    # Token not found in prompt_logprobs, use -1 as placeholder
                    prompt_token_ids.append(-1)
            else:
                # First token (position 0) has null prompt_logprobs
                # We can't determine its ID from this response
                prompt_token_ids.append(-1)

        if not text_offset:
            raise RuntimeError(
                "vLLM response missing text_offset. "
                "This may be due to an older vLLM version or server configuration. "
                "text_offset is required for text-based log-prob scoring."
            )

        # Find the first token that starts at or after the continuation
        # This handles cases where context ends mid-token
        cont_start_idx = None
        for i, offset in enumerate(text_offset):
            if offset >= continuation_start_char:
                cont_start_idx = i
                break

        if cont_start_idx is None:
            # All tokens are context, no continuation tokens
            return TextScoringResult(
                token_ids=tuple(prompt_token_ids),
                logprobs=tuple(token_logprobs),
                answer_range=(len(prompt_token_ids), len(prompt_token_ids)),
                total_logprob=0.0,
                request_id=request_id,
            )

        # Sum logprobs for continuation tokens
        # Note: token_logprobs[0] is typically None (no prior context for first token)
        continuation_logprobs = token_logprobs[cont_start_idx:]
        valid_logprobs = [lp for lp in continuation_logprobs if lp is not None]
        total_logprob = sum(valid_logprobs)

        return TextScoringResult(
            token_ids=tuple(prompt_token_ids),
            logprobs=tuple(token_logprobs),
            answer_range=(cont_start_idx, len(prompt_token_ids)),
            total_logprob=total_logprob,
            request_id=request_id,
        )

