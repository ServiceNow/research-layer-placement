"""Generation backends for trace pre-computation.

Provides a categorical abstraction for generation backends:
- AsyncBackend: For remote servers (vLLM, sglang, OpenAI-compatible APIs)
- SyncBackend: For local models (transformers)

The command uses different execution strategies based on category:
- AsyncBackend → async worker pool with max_concurrent_requests
- SyncBackend → sync loop (sequential or batched processing)
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Self, Sequence, TypedDict

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from place_layers.evaluator.importance_weighted.apriel_validation import (
        Apriel2Structure,
    )
    from place_layers.evaluator.importance_weighted.vllm_client import GenerationResult


class ChatMessage(TypedDict):
    """A single message in a chat conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


# =============================================================================
# Base Classes
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
    async def compute_batch_log_probs(
        self,
        traces: Sequence[tuple[Sequence[int], Sequence[int]]],
    ) -> list[float]:
        """Compute log probabilities for a batch of traces.

        Args:
            traces: List of (prompt_tokens, completion_tokens) tuples.

        Returns:
            List of log probabilities (sum over completion tokens) in input order.
        """
        pass

    @abstractmethod
    async def set_placement(self, mixer_names: Sequence[str]) -> None:
        """Set the mixer placement for each layer.

        Args:
            mixer_names: Mixer name for each layer (e.g., ["attention", "mamba", ...]).
                        Length must match number of layers with stochastic mixers.

        Raises:
            NotImplementedError: If backend doesn't support dynamic placement.
            ValueError: If model is not a supported supernet architecture.
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
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        request_id: str = "",
    ) -> "GenerationResult":
        """Generate n completions for a single message sequence.

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
    def batch_generate(
        self,
        messages_batch: Sequence[list[ChatMessage]],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list["GenerationResult"]:
        """Generate n completions for each message sequence in batch.

        Args:
            messages_batch: List of message sequences.
            n: Number of completions per sequence.
            max_tokens: Maximum tokens per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.

        Returns:
            List of GenerationResult, one per message sequence.
        """
        pass

    @abstractmethod
    def compute_batch_log_probs(
        self,
        traces: Sequence[tuple[Sequence[int], Sequence[int]]],
    ) -> list[float]:
        """Compute log probabilities for a batch of traces.

        Uses config.batch_size for internal batching.

        Args:
            traces: List of (prompt_tokens, completion_tokens) tuples.

        Returns:
            List of log probabilities (sum over completion tokens) in input order.
        """
        pass

    @abstractmethod
    def set_placement(self, mixer_names: Sequence[str]) -> None:
        """Set the mixer placement for each layer.

        Args:
            mixer_names: Mixer name for each layer (e.g., ["attention", "mamba", ...]).
                        Length must match number of layers with stochastic mixers.

        Raises:
            NotImplementedError: If backend doesn't support dynamic placement.
            ValueError: If model is not a supported supernet architecture.
        """
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


# =============================================================================
# vLLM Backend (Async)
# =============================================================================


class VLLMBackend(AsyncBackend):
    """Async generation backend using vLLM server."""

    def __init__(self, config: "VLLMBackendConfig") -> None:
        from place_layers.evaluator.importance_weighted.vllm_client import (
            VLLMClient,
            VLLMClientConfig,
        )

        self._config = config
        vllm_config = VLLMClientConfig(
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
            max_concurrent_requests=config.max_concurrent_requests,
        )
        self._client = VLLMClient(vllm_config)

    @property
    def max_concurrent_requests(self) -> int:
        """Maximum concurrent requests from config."""
        return self._config.max_concurrent_requests

    async def __aenter__(self) -> Self:
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._client.__aexit__(exc_type, exc_val, exc_tb)

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
        return await self._client.chat_generate(
            messages=messages,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            request_id=request_id,
        )

    async def compute_batch_log_probs(
        self,
        traces: Sequence[tuple[Sequence[int], Sequence[int]]],
    ) -> list[float]:
        """Compute log probabilities using vLLM's echo mode.

        Sends all scoring requests concurrently, bounded by the client's semaphore.
        """
        if not traces:
            return []

        tasks = [
            self._client.score_completion(prompt, completion, request_id=str(i))
            for i, (prompt, completion) in enumerate(traces)
        ]

        results = await asyncio.gather(*tasks)
        return [r.total_logprob for r in results]

    async def set_placement(self, mixer_names: Sequence[str]) -> None:
        """Set mixer placement - NOT YET IMPLEMENTED for vLLM.

        vLLM backend requires custom model integration to support
        dynamic mixer placement. This will be implemented when the
        Apriel2 model is ported to vLLM with placement switching support.
        """
        raise NotImplementedError(
            "Dynamic mixer placement is not yet supported for vLLM backend. "
            "Use TransformersBackend for placement switching, or implement "
            "vLLM model integration with /v1/set_placement endpoint."
        )


class VLLMBackendConfig(BaseBackendConfig):
    """Configuration for vLLM generation backend."""

    _backend_class: ClassVar[type[AsyncBackend]] = VLLMBackend

    backend_type: Literal["vllm"] = "vllm"

    base_url: str = Field(description="vLLM server base URL")
    model: str = Field(description="Model name as registered in vLLM")
    api_key: str = Field(default="EMPTY", description="API key for vLLM")
    timeout_seconds: int = Field(default=300, description="Request timeout")
    max_concurrent_requests: int = Field(
        default=32, description="Max concurrent HTTP requests"
    )


# =============================================================================
# Transformers Backend (Sync)
# =============================================================================


class TransformersBackend(SyncBackend):
    """Sync generation backend using HuggingFace Transformers.

    Wraps TransformersClient for model lifecycle and generation operations.
    Supports dynamic mixer placement for Apriel2 supernet models.
    """

    def __init__(self, config: "TransformersBackendConfig") -> None:
        from place_layers.evaluator.importance_weighted.transformers_client import (
            TransformersClient,
            TransformersClientConfig,
        )

        self._config = config
        client_config = TransformersClientConfig(
            model_path=config.model_path,
            tokenizer_path=config.tokenizer_path,
            torch_dtype=config.torch_dtype,
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code,
        )
        self._client = TransformersClient(client_config)

        # Lazy-initialized on first set_placement call
        self._apriel_structure: "Apriel2Structure | None" = None
        self._current_placement: tuple[str, ...] | None = None

    @property
    def batch_size(self) -> int:
        """Number of prompts to batch together."""
        return self._config.batch_size

    def __enter__(self) -> Self:
        self._client.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._client.__exit__(exc_type, exc_val, exc_tb)

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        request_id: str = "",
    ) -> "GenerationResult":
        """Generate n completions for a single message sequence."""
        results = self._client.generate_batch(
            [messages],
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        result = results[0]
        # Set request_id on the result
        return type(result)(
            request_id=request_id,
            samples=result.samples,
            prompt_token_ids=result.prompt_token_ids,
        )

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
        return self._client.generate_batch(
            list(messages_batch),
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def compute_batch_log_probs(
        self,
        traces: Sequence[tuple[Sequence[int], Sequence[int]]],
    ) -> list[float]:
        """Compute log probabilities using transformers forward passes.

        Delegates to client's smart batching implementation which groups
        traces by length for efficient GPU utilization. Uses config.batch_size.
        """
        # Convert to the format expected by TransformersClient
        trace_tuples = [
            (tuple(prompt), tuple(completion)) for prompt, completion in traces
        ]
        return self._client.compute_batch_log_probs(
            trace_tuples, batch_size=self._config.batch_size
        )

    def set_placement(self, mixer_names: Sequence[str]) -> None:
        """Set mixer placement for Apriel2 supernet model.

        On first call, validates the model is an Apriel2 supernet with
        stochastic mixers. Then sets the main_mixer_name for each layer.

        Args:
            mixer_names: Mixer name for each layer (e.g., ["attention", "mamba", ...]).

        Raises:
            ValueError: If model is not Apriel2 or mixer_names length doesn't match.
        """
        from place_layers.evaluator.importance_weighted.apriel_validation import (
            Apriel2ValidationError,
            validate_apriel2_model,
        )

        placement_tuple = tuple(mixer_names)

        # Skip if already set to this placement
        if self._current_placement == placement_tuple:
            return

        # Lazy validation on first call
        if self._apriel_structure is None:
            try:
                self._apriel_structure = validate_apriel2_model(self._client.model)
            except Apriel2ValidationError as e:
                raise ValueError(
                    f"Model does not support dynamic placement: {e}"
                ) from e

        # Validate length matches
        if len(mixer_names) != self._apriel_structure.num_layers:
            raise ValueError(
                f"mixer_names length ({len(mixer_names)}) doesn't match "
                f"model layers ({self._apriel_structure.num_layers})"
            )

        # Validate all mixer names are valid
        valid_mixers = set(self._apriel_structure.mixer_names)
        for layer_idx, mixer_name in enumerate(mixer_names):
            if mixer_name not in valid_mixers:
                raise ValueError(
                    f"Invalid mixer '{mixer_name}' at layer {layer_idx}. "
                    f"Valid mixers: {valid_mixers}"
                )

        # Set the placement on each layer's stochastic mixer
        model = self._client.model
        for layer_idx, mixer_name in enumerate(mixer_names):
            model.model.decoder.blocks[layer_idx].mixer.main_mixer_name = mixer_name

        self._current_placement = placement_tuple


class TransformersBackendConfig(BaseBackendConfig):
    """Configuration for Transformers generation backend."""

    _backend_class: ClassVar[type[SyncBackend]] = TransformersBackend

    backend_type: Literal["transformers"] = "transformers"

    model_path: str = Field(description="Path to model (local or HF hub)")
    tokenizer_path: str | None = Field(
        default=None,
        description="Path to tokenizer (defaults to model_path)",
    )
    torch_dtype: Literal["float16", "bfloat16", "float32"] = Field(
        default="bfloat16",
        description="Model dtype",
    )
    device_map: str = Field(
        default="cuda",
        description="Device map for model loading",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Trust remote code when loading model",
    )
    batch_size: int = Field(
        default=1,
        description="Number of prompts to batch together for generation",
        gt=0,
    )


# =============================================================================
# Discriminated Union
# =============================================================================

# Import REST backend here to avoid circular imports (it imports from this module)
from place_layers.evaluator.importance_weighted.rest_backend import RESTBackendConfig

BackendConfig = Annotated[
    VLLMBackendConfig | TransformersBackendConfig | RESTBackendConfig,
    Field(discriminator="backend_type"),
]
