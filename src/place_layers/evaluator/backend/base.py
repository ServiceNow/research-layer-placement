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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Self, Sequence, TypedDict

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from place_layers.evaluator.backend.apriel_validation import (
        Apriel2Structure,
    )
    from place_layers.evaluator.backend.vllm import GenerationResult


class ChatMessage(TypedDict):
    """A single message in a chat conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


# =============================================================================
# Model Structure
# =============================================================================


@dataclass(frozen=True)
class ModelStructure:
    """Model structure for placement optimization.

    Generic representation of model structure that backends expose.
    Decoupled from Apriel2-specific details.
    """

    num_layers: int
    """Number of layers with switchable mixers."""

    mixer_names: tuple[str, ...]
    """Available mixer types (e.g., ('attention', 'gdn', 'kda'))."""


@dataclass(frozen=True)
class ChatLogProbResult:
    """Detailed result from chat log-prob computation.

    Contains all token-level information needed for observability:
    - Full token sequence (after chat template applied)
    - Per-token logprobs
    - Answer range (which tokens were identified as "the answer")
    - Total score (sum of answer logprobs)
    """

    token_ids: tuple[int, ...]
    """Full token sequence for this chat (context + completion)."""

    logprobs: tuple[float | None, ...]
    """Per-token logprobs (None for first token which has no prior context)."""

    answer_range: tuple[int, int]
    """(start, end) indices identifying answer tokens in token_ids."""

    total_log_prob: float
    """Sum of logprobs for answer tokens (used for prediction)."""


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
        """Compute log probabilities for a batch of traces (token-based).

        Args:
            traces: List of (prompt_tokens, completion_tokens) tuples.

        Returns:
            List of log probabilities (sum over completion tokens) in input order.
        """
        pass

    @abstractmethod
    async def compute_text_log_probs(
        self,
        contexts: Sequence[str],
        continuations: Sequence[str],
    ) -> list[float]:
        """Compute log probabilities for text continuations (text-based).

        This is the text-based counterpart to compute_batch_log_probs.
        Tokenization is handled internally by the backend.

        Args:
            contexts: List of context strings (prompts).
            continuations: List of continuation strings to score.

        Returns:
            List of log P(continuation | context) values in input order.

        Raises:
            NotImplementedError: If backend doesn't support text-based scoring.
        """
        pass

    @abstractmethod
    async def compute_chat_log_probs(
        self,
        chats: Sequence[tuple[Sequence[ChatMessage], Sequence[ChatMessage]]],
    ) -> list[ChatLogProbResult]:
        """Compute log probabilities for chat completions with full observability.

        This is the chat-level counterpart to compute_batch_log_probs.
        Each chat is a (context_messages, completion_messages) tuple.

        Args:
            chats: List of (context_messages, completion_messages) tuples.
                context_messages: Messages forming the context (e.g., few-shot + question).
                completion_messages: Messages to score (e.g., assistant answer).

        Returns:
            List of ChatLogProbResult with token IDs, per-token logprobs,
            answer ranges, and total scores for each chat.
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

    @property
    @abstractmethod
    def model_structure(self) -> ModelStructure:
        """Model structure (num_layers, mixer_names).

        Available after entering async context (__aenter__).

        Raises:
            NotImplementedError: If backend doesn't support model structure discovery.
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
        """Compute log probabilities for a batch of traces (token-based).

        Uses config.batch_size for internal batching.

        Args:
            traces: List of (prompt_tokens, completion_tokens) tuples.

        Returns:
            List of log probabilities (sum over completion tokens) in input order.
        """
        pass

    @abstractmethod
    def compute_text_log_probs(
        self,
        contexts: Sequence[str],
        continuations: Sequence[str],
    ) -> list[float]:
        """Compute log probabilities for text continuations (text-based).

        This is the text-based counterpart to compute_batch_log_probs.
        Tokenization is handled internally by the backend.

        Args:
            contexts: List of context strings (prompts).
            continuations: List of continuation strings to score.

        Returns:
            List of log P(continuation | context) values in input order.

        Raises:
            NotImplementedError: If backend doesn't support text-based scoring.
        """
        pass

    @abstractmethod
    def compute_chat_log_probs(
        self,
        chats: Sequence[tuple[Sequence[ChatMessage], Sequence[ChatMessage]]],
    ) -> list[ChatLogProbResult]:
        """Compute log probabilities for chat completions with full observability.

        This is the chat-level counterpart to compute_batch_log_probs.
        Each chat is a (context_messages, completion_messages) tuple.

        Args:
            chats: List of (context_messages, completion_messages) tuples.
                context_messages: Messages forming the context (e.g., few-shot + question).
                completion_messages: Messages to score (e.g., assistant answer).

        Returns:
            List of ChatLogProbResult with token IDs, per-token logprobs,
            answer ranges, and total scores for each chat.
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

    @property
    @abstractmethod
    def model_structure(self) -> ModelStructure:
        """Model structure (num_layers, mixer_names).

        Available after entering context (__enter__).

        Raises:
            NotImplementedError: If backend doesn't support model structure discovery.
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
        from place_layers.evaluator.backend.vllm import (
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

    async def compute_text_log_probs(
        self,
        contexts: Sequence[str],
        continuations: Sequence[str],
    ) -> list[float]:
        """Compute log probabilities for text continuations.

        Uses vLLM's text_offset in the response to determine where the
        continuation starts, avoiding the need for local tokenization.

        Args:
            contexts: List of context strings (prompts).
            continuations: List of continuation strings to score.

        Returns:
            List of log P(continuation | context) values in input order.
        """
        if len(contexts) != len(continuations):
            raise ValueError(
                f"contexts and continuations must have same length, "
                f"got {len(contexts)} and {len(continuations)}"
            )

        if not contexts:
            return []

        # Send all scoring requests concurrently, bounded by semaphore
        tasks = [
            self._client.score_text_completion(ctx, cont, request_id=str(i))
            for i, (ctx, cont) in enumerate(zip(contexts, continuations))
        ]

        results = await asyncio.gather(*tasks)
        return [r.total_logprob for r in results]

    async def compute_chat_log_probs(
        self,
        chats: Sequence[tuple[Sequence[ChatMessage], Sequence[ChatMessage]]],
    ) -> list[ChatLogProbResult]:
        """Compute log probabilities for chat completions with full observability.

        Uses vLLM's prompt_logprobs feature with a diff-based approach:
        1. Group chats by context (same context = same multiple-choice question)
        2. Send requests with prompt_logprobs=1 for each chat
        3. Compare token IDs across chats in each group
        4. Find differing positions (these are the answer tokens)
        5. Sum logprobs at differing positions

        This approach is template-agnostic - we don't need to know the chat
        template structure, just find where tokens differ.

        Args:
            chats: List of (context_messages, completion_messages) tuples.

        Returns:
            List of ChatLogProbResult with full token-level detail.
        """
        if not chats:
            return []

        from collections import defaultdict

        # Group chats by context for diff-based scoring
        # context_key -> [(original_idx, completion_msgs)]
        context_groups: dict[tuple, list[tuple[int, Sequence[ChatMessage]]]] = (
            defaultdict(list)
        )
        context_msgs_by_key: dict[tuple, Sequence[ChatMessage]] = {}

        for i, (ctx_msgs, comp_msgs) in enumerate(chats):
            ctx_key = tuple((m["role"], m["content"]) for m in ctx_msgs)
            context_groups[ctx_key].append((i, comp_msgs))
            context_msgs_by_key[ctx_key] = ctx_msgs

        # Process each context group
        all_results: list[ChatLogProbResult | None] = [None] * len(chats)

        for ctx_key, group in context_groups.items():
            ctx_msgs = context_msgs_by_key[ctx_key]
            results = await self._score_chat_group(ctx_msgs, group)
            for (orig_idx, _), result in zip(group, results):
                all_results[orig_idx] = result

        # Fill any None results with empty results (shouldn't happen)
        return [
            r if r is not None else ChatLogProbResult(
                token_ids=(), logprobs=(), answer_range=(0, 0), total_log_prob=0.0
            )
            for r in all_results
        ]

    async def _score_chat_group(
        self,
        context_msgs: Sequence[ChatMessage],
        group: list[tuple[int, Sequence[ChatMessage]]],
    ) -> list[ChatLogProbResult]:
        """Score a group of chats that share the same context.

        Uses diff-based approach: find positions where token IDs differ,
        then sum logprobs at those positions.

        Returns ChatLogProbResult with full token-level information.
        """
        session = self._client._ensure_session()
        url = f"{self._client.config.base_url.rstrip('/')}/v1/chat/completions"

        # Send all requests in parallel
        async def send_request(
            comp_msgs: Sequence[ChatMessage],
        ) -> dict:
            messages = list(context_msgs) + list(comp_msgs)
            payload = {
                "model": self._client.config.model,
                "messages": messages,
                "max_tokens": 1,
                "prompt_logprobs": 1,
                "temperature": 0,
            }
            async with self._client._semaphore:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(
                            f"vLLM chat request failed: {resp.status}: {error_text}"
                        )
                    return await resp.json()

        responses = await asyncio.gather(*[send_request(comp) for _, comp in group])

        # Extract token IDs and logprobs from each response
        token_sequences: list[list[int | None]] = []
        logprob_sequences: list[list[float | None]] = []

        for data in responses:
            prompt_logprobs = data.get("prompt_logprobs", [])
            tokens: list[int | None] = []
            logprobs: list[float | None] = []

            for lp in prompt_logprobs:
                if lp is None:
                    tokens.append(None)
                    logprobs.append(None)
                else:
                    # The key is the actual token ID (as string)
                    token_id = int(list(lp.keys())[0])
                    token_logprob = lp[list(lp.keys())[0]]["logprob"]
                    tokens.append(token_id)
                    logprobs.append(token_logprob)

            token_sequences.append(tokens)
            logprob_sequences.append(logprobs)

        # Find answer token ranges for each sequence (excluding common prefix/suffix)
        from place_layers.evaluator.lm_eval_utils import find_differing_ranges

        answer_ranges = find_differing_ranges(token_sequences)

        # Build ChatLogProbResult for each chat
        results: list[ChatLogProbResult] = []
        for tokens, logprobs, (start, end) in zip(
            token_sequences, logprob_sequences, answer_ranges
        ):
            # Extract answer logprobs and compute sum
            answer_logprobs: list[float] = [
                logprobs[pos]  # type: ignore[misc]
                for pos in range(start, end)
                if pos < len(logprobs) and logprobs[pos] is not None
            ]
            total_log_prob = sum(answer_logprobs) if answer_logprobs else 0.0

            # Convert token_ids: replace None with 0 (shouldn't happen in practice)
            clean_token_ids = tuple(t if t is not None else 0 for t in tokens)

            results.append(ChatLogProbResult(
                token_ids=clean_token_ids,
                logprobs=tuple(logprobs),
                answer_range=(start, end),
                total_log_prob=total_log_prob,
            ))

        return results

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

    @property
    def model_structure(self) -> ModelStructure:
        """Model structure - NOT SUPPORTED for vLLM.

        vLLM doesn't expose supernet model structure. Use TransformersBackend
        or RESTBackend for operations requiring model structure.
        """
        raise NotImplementedError(
            "Model structure discovery is not supported for vLLM backend. "
            "Use TransformersBackend or RESTBackend for evaluate/optimize commands."
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
        from place_layers.evaluator.backend.transformers import (
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
        from place_layers.evaluator.backend.apriel_validation import (
            Apriel2ValidationError,
            validate_apriel2_model,
        )

        self._client.__enter__()

        # Eagerly validate model structure on entry
        try:
            self._apriel_structure = validate_apriel2_model(self._client.model)
        except Apriel2ValidationError:
            # Not an Apriel2 supernet - leave structure as None
            # This allows using TransformersBackend for non-supernet models
            # (e.g., precompute_traces with Qwen2)
            pass

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._client.__exit__(exc_type, exc_val, exc_tb)

    @property
    def model_structure(self) -> ModelStructure:
        """Model structure (num_layers, mixer_names).

        Available after entering context (__enter__).

        Raises:
            NotImplementedError: If model is not an Apriel2 supernet.
        """
        if self._apriel_structure is None:
            raise NotImplementedError(
                "Model structure discovery requires an Apriel2 supernet model. "
                "The loaded model does not support dynamic mixer placement."
            )
        return ModelStructure(
            num_layers=self._apriel_structure.num_layers,
            mixer_names=self._apriel_structure.mixer_names,
        )

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

    def compute_text_log_probs(
        self,
        contexts: Sequence[str],
        continuations: Sequence[str],
    ) -> list[float]:
        """Compute log probabilities for text continuations.

        Tokenizes contexts and continuations internally, then delegates to
        compute_batch_log_probs for efficient batched computation.

        Args:
            contexts: List of context strings (prompts).
            continuations: List of continuation strings to score.

        Returns:
            List of log P(continuation | context) values in input order.
        """
        if len(contexts) != len(continuations):
            raise ValueError(
                f"contexts and continuations must have same length, "
                f"got {len(contexts)} and {len(continuations)}"
            )

        if not contexts:
            return []

        tokenizer = self._client.tokenizer

        # Tokenize all contexts and continuations
        traces: list[tuple[Sequence[int], Sequence[int]]] = []
        for ctx, cont in zip(contexts, continuations):
            # Context gets special tokens (BOS), continuation does not
            ctx_tokens = tokenizer.encode(ctx, add_special_tokens=True)
            cont_tokens = tokenizer.encode(cont, add_special_tokens=False)
            traces.append((ctx_tokens, cont_tokens))

        # Delegate to token-based method
        return self.compute_batch_log_probs(traces)

    def compute_chat_log_probs(
        self,
        chats: Sequence[tuple[Sequence[ChatMessage], Sequence[ChatMessage]]],
    ) -> list[ChatLogProbResult]:
        """Compute log probabilities for chat completions with full observability.

        Uses prefix/suffix matching approach (same as vLLM backend):
        1. Group chats by context (same context = same multiple-choice question)
        2. Tokenize all full messages for each group
        3. Find answer token ranges using prefix/suffix matching
        4. Compute logprobs via forward pass
        5. SUM logprobs for answer tokens (matches lm-eval semantics)

        This approach is template-agnostic - we find common prefix (context +
        template overhead) and common suffix (end tokens) to isolate the
        actual answer tokens.

        Args:
            chats: List of (context_messages, completion_messages) tuples.

        Returns:
            List of ChatLogProbResult with full token-level detail.
        """
        if not chats:
            return []

        from collections import defaultdict

        tokenizer = self._client.tokenizer

        # Group chats by context for diff-based scoring
        context_groups: dict[tuple, list[tuple[int, Sequence[ChatMessage]]]] = (
            defaultdict(list)
        )
        context_msgs_by_key: dict[tuple, Sequence[ChatMessage]] = {}

        for i, (ctx_msgs, comp_msgs) in enumerate(chats):
            ctx_key = tuple((m["role"], m["content"]) for m in ctx_msgs)
            context_groups[ctx_key].append((i, comp_msgs))
            context_msgs_by_key[ctx_key] = ctx_msgs

        # Process each context group
        all_results: list[ChatLogProbResult | None] = [None] * len(chats)

        for ctx_key, group in context_groups.items():
            ctx_msgs = context_msgs_by_key[ctx_key]

            # Tokenize all full messages in this group
            token_sequences: list[list[int]] = []
            for _, comp_msgs in group:
                full_msgs = list(ctx_msgs) + list(comp_msgs)
                full_text = tokenizer.apply_chat_template(
                    full_msgs,  # type: ignore[arg-type]
                    tokenize=False,
                    add_generation_prompt=False,
                )
                tokens = tokenizer.encode(full_text, add_special_tokens=False)  # type: ignore[arg-type]
                token_sequences.append(tokens)

            # Find answer token ranges using prefix/suffix matching
            from place_layers.evaluator.lm_eval_utils import find_differing_ranges

            answer_ranges = find_differing_ranges(token_sequences)  # type: ignore[arg-type]

            # Compute logprobs and build results
            results = self._score_sequences_in_ranges(token_sequences, answer_ranges)

            for (orig_idx, _), result in zip(group, results):
                all_results[orig_idx] = result

        # Fill any None results with empty results (shouldn't happen)
        return [
            r if r is not None else ChatLogProbResult(
                token_ids=(), logprobs=(), answer_range=(0, 0), total_log_prob=0.0
            )
            for r in all_results
        ]

    def _score_sequences_in_ranges(
        self,
        token_sequences: list[list[int]],
        ranges: list[tuple[int, int]],
    ) -> list[ChatLogProbResult]:
        """Compute logprobs for each sequence and return full results.

        Args:
            token_sequences: List of token ID sequences.
            ranges: List of (start, end) tuples, one per sequence.

        Returns:
            List of ChatLogProbResult with token IDs, per-token logprobs,
            answer ranges, and total scores.
        """
        import torch

        model = self._client.model
        device = next(model.parameters()).device

        results: list[ChatLogProbResult] = []

        for tokens, (start, end) in zip(token_sequences, ranges):
            if not tokens:
                results.append(ChatLogProbResult(
                    token_ids=(),
                    logprobs=(),
                    answer_range=(start, end),
                    total_log_prob=0.0,
                ))
                continue

            # Run forward pass
            input_ids = torch.tensor([tokens], device=device)
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits[0]  # [seq_len, vocab_size]

            # Compute log probs for each token position
            log_probs_tensor = torch.log_softmax(logits, dim=-1)

            # Build per-token logprobs list
            # logprobs[i] is the logprob of token[i] given tokens[:i]
            # First token has no prior context, so it's None
            all_logprobs: list[float | None] = [None]
            for pos in range(1, len(tokens)):
                token_id = tokens[pos]
                lp = log_probs_tensor[pos - 1, token_id].item()
                all_logprobs.append(lp)

            # Extract answer logprobs and compute sum
            answer_logprobs: list[float] = [
                all_logprobs[pos]  # type: ignore[misc]
                for pos in range(start, end)
                if pos < len(all_logprobs) and all_logprobs[pos] is not None
            ]
            total_log_prob = sum(answer_logprobs) if answer_logprobs else 0.0

            results.append(ChatLogProbResult(
                token_ids=tuple(tokens),
                logprobs=tuple(all_logprobs),
                answer_range=(start, end),
                total_log_prob=total_log_prob,
            ))

        return results

    def set_placement(self, mixer_names: Sequence[str]) -> None:
        """Set mixer placement for Apriel2 supernet model.

        Args:
            mixer_names: Mixer name for each layer (e.g., ["attention", "mamba", ...]).

        Raises:
            ValueError: If model is not Apriel2 or mixer_names length doesn't match.
        """
        placement_tuple = tuple(mixer_names)

        # Skip if already set to this placement
        if self._current_placement == placement_tuple:
            return

        # Require Apriel2 supernet (validated in __enter__)
        if self._apriel_structure is None:
            raise ValueError(
                "Model does not support dynamic placement. "
                "Only Apriel2 supernet models support set_placement()."
            )

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
from place_layers.evaluator.backend.rest import RESTBackendConfig

BackendConfig = Annotated[
    VLLMBackendConfig | TransformersBackendConfig | RESTBackendConfig,
    Field(discriminator="backend_type"),
]
