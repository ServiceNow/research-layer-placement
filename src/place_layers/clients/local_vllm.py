"""Async client for local vLLM using in-process engine with OpenAI serving layer.

Same interface as RemoteVLLMClient but runs vLLM in-process.
Uses vLLM's OpenAIServingChat and OpenAIServingCompletion for high-level API.
Supports Apriel2 models with runtime mixer placement switching.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING, Literal

import torch
import triton
from pydantic import BaseModel, Field

from place_layers.core.types import ChatMessage
from place_layers.clients.base_vllm import BaseVLLMClient
from place_layers.clients.remote_vllm import VLLMBadRequestError
from place_layers.clients.types import (
    GeneratedSample,
    GenerationResult,
    ScoringResult,
    SequenceLogProbs,
)

if TYPE_CHECKING:
    from typing import Self

    from vllm.entrypoints.openai.protocol import ErrorResponse
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
    from vllm.v1.engine.async_llm import AsyncLLM

logger = logging.getLogger(__name__)


# Triton allocator setup - must be done lazily to avoid initializing CUDA
# before vLLM can spawn its engine process
_triton_allocator_set = False


def _ensure_triton_allocator() -> None:
    """Set triton allocator lazily to avoid early CUDA init."""
    global _triton_allocator_set
    if _triton_allocator_set:
        return

    def _triton_allocator(size: int, alignment: int, stream: int | None):  # type: ignore[return]
        return torch.empty(size, dtype=torch.int8, device="cuda").data_ptr()

    triton.set_allocator(_triton_allocator)
    _triton_allocator_set = True


class LocalVLLMClientConfig(BaseModel):
    """Configuration for local vLLM client."""

    model_config = {"frozen": True}

    client_type: Literal["local"] = Field(default="local", description="Client type discriminator")
    model_path: str = Field(description="Path to the model checkpoint")
    tokenizer_path: str | None = Field(
        default=None, description="Path to tokenizer (defaults to model_path)"
    )
    torch_dtype: Literal["float16", "bfloat16", "float32", "auto"] = Field(
        description="Model dtype"
    )
    trust_remote_code: bool = Field(description="Trust remote code for model loading")
    gpu_memory_utilization: float = Field(description="Fraction of GPU memory to use")
    max_model_len: int | None = Field(
        default=None, description="Maximum model context length"
    )
    disable_compile: bool = Field(description="Disable torch.compile")
    enable_prefix_caching: bool = Field(description="Enable prefix caching")
    disable_flash_attn: bool = Field(
        default=False, description="Disable Flash Attention (use xformers instead)"
    )
    max_concurrent_requests: int = Field(description="Maximum concurrent requests")


class LocalVLLMClient(BaseVLLMClient):
    """Async client using in-process vLLM with OpenAI-compatible serving layer.

    Uses vLLM's OpenAIServingChat and OpenAIServingCompletion internally,
    providing the same high-level API as the HTTP server but without network overhead.

    Same interface as RemoteVLLMClient.
    Supports Apriel2 models with runtime mixer placement switching.

    Must be used as an async context manager:

        async with LocalVLLMClient(config) as client:
            result = await client.chat_generate(...)
    """

    def __init__(self, config: LocalVLLMClientConfig) -> None:
        self.config = config
        self._engine: "AsyncLLM | None" = None
        self._chat_serving: "OpenAIServingChat | None" = None
        self._completion_serving: "OpenAIServingCompletion | None" = None
        self._model_name: str = ""
        self._supports_placement = False
        self._num_layers: int | None = None
        self._mixer_names: list[str] | None = None
        self._current_placement: list[str] | None = None

    async def __aenter__(self) -> "Self":
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
        from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
        from vllm.entrypoints.openai.serving_models import (
            BaseModelPath,
            OpenAIServingModels,
        )
        from vllm.v1.engine.async_llm import AsyncLLM

        # Enable verbose vLLM logging to capture subprocess errors
        # These must be set before creating the engine subprocess
        os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
        os.environ["VLLM_CONFIGURE_LOGGING"] = "1"

        # Redirect stdout/stderr to capture vLLM subprocess output
        # This helps capture crashes from the EngineCore subprocess
        vllm_log_file = "vllm_engine.log"
        logger.info(f"vLLM subprocess output will be captured to: {vllm_log_file}")

        # Save original stdout/stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        try:
            # Open log file for vLLM subprocess output
            vllm_log = open(vllm_log_file, "a", buffering=1)  # Line buffered

            # Redirect stdout/stderr
            sys.stdout = vllm_log
            sys.stderr = vllm_log

            logger.info(f"Loading vLLM model from {self.config.model_path}")
            logger.info(f"vLLM config: max_model_len={self.config.max_model_len}, "
                       f"gpu_memory_utilization={self.config.gpu_memory_utilization}, "
                       f"dtype={self.config.torch_dtype}")

            engine_args = AsyncEngineArgs(
                model=self.config.model_path,
                tokenizer=self.config.tokenizer_path or self.config.model_path,
                trust_remote_code=self.config.trust_remote_code,
                dtype=self.config.torch_dtype,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                disable_log_stats=False,  # Enable stats to see what's happening
                enable_prefix_caching=self.config.enable_prefix_caching,
                enforce_eager=self.config.disable_compile,
                disable_flash_attn=self.config.disable_flash_attn,
            )

            # Create async engine
            logger.info("Creating vLLM AsyncLLM engine (this will spawn a subprocess)...")
            self._engine = AsyncLLM.from_engine_args(engine_args)
            logger.info("vLLM engine subprocess spawned successfully")

        finally:
            # Restore original stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            # Keep file open - will be closed on exit
        self._model_name = self.config.model_path

        # Set triton allocator after engine spawns to avoid CUDA init in parent
        _ensure_triton_allocator()

        # Create OpenAI serving layer
        base_model_paths = [BaseModelPath(name=self._model_name, model_path=self.config.model_path)]

        serving_models = OpenAIServingModels(
            engine_client=self._engine,
            base_model_paths=base_model_paths,
        )

        self._chat_serving = OpenAIServingChat(
            engine_client=self._engine,
            models=serving_models,
            response_role="assistant",
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
            return_tokens_as_token_ids=True,
        )

        self._completion_serving = OpenAIServingCompletion(
            engine_client=self._engine,
            models=serving_models,
            request_logger=None,
            return_tokens_as_token_ids=True,
        )

        # Check if model supports placement switching (Apriel2 stochastic)
        try:
            placements = await self._engine.collective_rpc("get_layer_placements")
            if placements and placements[0]:
                self._supports_placement = True
                self._num_layers = len(placements[0])
                self._current_placement = list(placements[0].values())
                # Get available mixer names
                mixer_names = await self._engine.collective_rpc("get_mixer_names")
                if mixer_names and mixer_names[0]:
                    self._mixer_names = mixer_names[0]
                logger.info(
                    f"Model supports placement switching: {self._num_layers} layers, "
                    f"mixers: {self._mixer_names}"
                )
        except Exception as e:
            logger.debug(f"Model does not support placement switching: {e}")

        logger.info("vLLM async engine loaded successfully")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None
        self._chat_serving = None
        self._completion_serving = None
        torch.cuda.empty_cache()

    @property
    def supports_placement(self) -> bool:
        """Whether model supports placement switching."""
        return self._supports_placement

    @property
    def num_layers(self) -> int | None:
        """Number of layers with switchable mixers."""
        return self._num_layers

    @property
    def mixer_names(self) -> list[str] | None:
        """Available mixer types."""
        return self._mixer_names

    @property
    def current_placement(self) -> list[str] | None:
        """Current mixer placement."""
        return self._current_placement

    async def set_placement(self, placement: list[str]) -> None:
        """Set mixer placement via collective_rpc."""
        if not self._supports_placement:
            raise ValueError("Model does not support placement switching")
        if self._engine is None:
            raise RuntimeError("Client not initialized")
        await self._engine.collective_rpc("set_layer_placements", args=(placement,))
        self._current_placement = placement

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

        Uses vLLM's OpenAIServingChat internally.

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
        from vllm.entrypoints.openai.protocol import (
            ChatCompletionRequest,
            ChatCompletionResponse,
            ErrorResponse,
        )

        if self._chat_serving is None:
            raise RuntimeError("Client not initialized")

        request = ChatCompletionRequest(
            model=self._model_name,
            messages=[m.model_dump() for m in messages],  # type: ignore[arg-type]
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=True,
            top_logprobs=1,
        )

        response = await self._chat_serving.create_chat_completion(request)

        if isinstance(response, ErrorResponse):
            raise VLLMBadRequestError(f"Chat completion failed: {response.error}")

        # Type narrow to ChatCompletionResponse
        if not isinstance(response, ChatCompletionResponse):
            raise VLLMBadRequestError(f"Unexpected response type: {type(response)}")

        # Extract samples from response
        samples: list[GeneratedSample] = []

        for choice in response.choices:
            token_logprobs: list[float] = []
            if choice.logprobs is not None and choice.logprobs.content is not None:
                token_logprobs = [item.logprob for item in choice.logprobs.content]

            samples.append(
                GeneratedSample(
                    text=choice.message.content or "",
                    token_ids=choice.token_ids or [],
                    token_logprobs=token_logprobs,
                    truncated=choice.finish_reason == "length",
                )
            )

        return GenerationResult(
            key=key,
            samples=samples,
            prompt_token_ids=response.prompt_token_ids or [],
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
    ) -> GenerationResult:
        """Generate completions from a text prompt.

        Uses vLLM's OpenAIServingCompletion internally.

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
        from vllm.entrypoints.openai.protocol import (
            CompletionRequest,
            CompletionResponse,
            ErrorResponse,
        )

        if self._completion_serving is None:
            raise RuntimeError("Client not initialized")

        request = CompletionRequest(
            model=self._model_name,
            prompt=prompt,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=1,
        )

        response = await self._completion_serving.create_completion(request)

        if isinstance(response, ErrorResponse):
            raise VLLMBadRequestError(f"Completion failed: {response.error}")

        # Type narrow to CompletionResponse
        if not isinstance(response, CompletionResponse):
            raise VLLMBadRequestError(f"Unexpected response type: {type(response)}")

        # Extract samples from response
        samples: list[GeneratedSample] = []

        for choice in response.choices:
            token_logprobs: list[float] = []
            if choice.logprobs is not None:
                token_logprobs = [
                    lp for lp in choice.logprobs.token_logprobs if lp is not None
                ]

            samples.append(
                GeneratedSample(
                    text=choice.text,
                    token_ids=choice.token_ids or [],
                    token_logprobs=token_logprobs,
                    truncated=choice.finish_reason == "length",
                )
            )

        # Get prompt token IDs from first choice
        prompt_token_ids: list[int] = []
        if response.choices and response.choices[0].prompt_token_ids:
            prompt_token_ids = response.choices[0].prompt_token_ids

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
        prompt_tokens: list[int],
        completion_tokens: list[int],
        key: str,
    ) -> ScoringResult:
        """Score a completion given prompt tokens.

        Uses vLLM's OpenAIServingCompletion with echo mode.

        Args:
            prompt_tokens: Token IDs for the prompt.
            completion_tokens: Token IDs for the completion.
            key: Request key for joining with input.

        Returns:
            ScoringResult with total log probability.
        """
        from vllm.entrypoints.openai.protocol import (
            CompletionRequest,
            CompletionResponse,
            ErrorResponse,
        )

        if self._completion_serving is None:
            raise RuntimeError("Client not initialized")

        full_tokens = list(prompt_tokens) + list(completion_tokens)

        request = CompletionRequest(
            model=self._model_name,
            prompt=full_tokens,
            max_tokens=0,
            echo=True,
            logprobs=1,
        )

        response = await self._completion_serving.create_completion(request)

        if isinstance(response, ErrorResponse):
            raise VLLMBadRequestError(f"Completion failed: {response.error}")
        if not isinstance(response, CompletionResponse):
            raise VLLMBadRequestError(f"Unexpected response type: {type(response)}")

        # Extract logprobs for completion tokens
        choice = response.choices[0]
        token_logprobs = []
        if choice.logprobs and choice.logprobs.token_logprobs:
            token_logprobs = choice.logprobs.token_logprobs

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
        from vllm.entrypoints.openai.protocol import (
            CompletionRequest,
            CompletionResponse,
            ErrorResponse,
        )

        if self._completion_serving is None:
            raise RuntimeError("Client not initialized")

        full_text = context + continuation

        request = CompletionRequest(
            model=self._model_name,
            prompt=full_text,
            max_tokens=0,
            echo=True,
            logprobs=1,
        )

        response = await self._completion_serving.create_completion(request)

        if isinstance(response, ErrorResponse):
            raise VLLMBadRequestError(f"Completion failed: {response.error}")
        if not isinstance(response, CompletionResponse):
            raise VLLMBadRequestError(f"Unexpected response type: {type(response)}")

        choice = response.choices[0]
        logprobs_data = choice.logprobs

        token_logprobs: list[float | None] = []

        if logprobs_data and logprobs_data.token_logprobs:
            token_logprobs = list(logprobs_data.token_logprobs)

        # Get token_ids from choice (CompletionResponseChoice.token_ids)
        # Fall back to placeholder if no token_ids but have logprobs tokens
        if choice.token_ids:
            token_ids = list(choice.token_ids)
        elif logprobs_data and logprobs_data.tokens:
            token_ids = [-1] * len(logprobs_data.tokens)
        else:
            token_ids = []

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
        from vllm.entrypoints.openai.protocol import (
            ChatCompletionRequest,
            ChatCompletionResponse,
            ErrorResponse,
        )

        if self._chat_serving is None:
            raise RuntimeError("Client not initialized")

        full_messages = context_messages + completion_messages

        # prompt_logprobs gives us logprobs for all input tokens
        # max_tokens=1 is the minimum vLLM accepts; generated token is discarded
        request = ChatCompletionRequest(
            model=self._model_name,
            messages=[m.model_dump() for m in full_messages],  # type: ignore[arg-type]
            max_tokens=1,
            prompt_logprobs=1,
            temperature=0,
        )

        response = await self._chat_serving.create_chat_completion(request)

        if isinstance(response, ErrorResponse):
            raise VLLMBadRequestError(f"Chat completion failed: {response.error}")
        if not isinstance(response, ChatCompletionResponse):
            raise VLLMBadRequestError(f"Unexpected response type: {type(response)}")

        # Extract prompt_logprobs from response
        prompt_logprobs_data = response.prompt_logprobs or []

        token_ids: list[int] = []
        logprobs: list[float | None] = []

        for lp in prompt_logprobs_data:
            if lp is None:
                token_ids.append(-1)
                logprobs.append(None)
            else:
                # lp is {token_id: Logprob} where Logprob has .logprob attribute
                token_id = int(list(lp.keys())[0])
                token_ids.append(token_id)
                logprobs.append(lp[token_id].logprob)

        return SequenceLogProbs(
            token_ids=token_ids,
            logprobs=logprobs,
            key=key,
        )
