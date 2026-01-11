"""REST backend for distributed TransformersBackend evaluation.

This module provides an AsyncBackend implementation that distributes
requests across multiple REST servers, each running a TransformersBackend.

The REST backend uses a placement-in-request pattern:
- set_placement() stores the placement locally
- All requests include the placement in their payload
- Servers set placement before processing (no-op if unchanged)

This design enables load balancing without coordination overhead.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, ClassVar, Literal, Self, Sequence

import aiohttp
from pydantic import Field

from place_layers.evaluator.importance_weighted.generation_backend import (
    AsyncBackend,
    BaseBackendConfig,
    ChatMessage,
    ModelStructure,
)

if TYPE_CHECKING:
    from place_layers.evaluator.importance_weighted.vllm_client import GenerationResult

logger = logging.getLogger(__name__)


class RESTBackendError(Exception):
    """Error communicating with REST server."""

    pass


class RESTBackend(AsyncBackend):
    """Async backend that distributes requests across multiple REST servers.

    Each server runs a TransformersBackend via the serve command. Requests
    are distributed using round-robin load balancing. The placement is
    included in every request, allowing servers to set it lazily.

    Usage:
        async with RESTBackend(config) as backend:
            await backend.set_placement(["attention", "mamba", ...])
            result = await backend.agenerate(messages, n=8, ...)
    """

    def __init__(self, config: "RESTBackendConfig") -> None:
        self._config = config
        self._servers = list(config.servers)
        self._session: aiohttp.ClientSession | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._idx = 0  # Round-robin index
        self._current_placement: list[str] | None = None
        self._model_structure: ModelStructure | None = None

        if not self._servers:
            raise ValueError("At least one server URL must be provided")

    @property
    def max_concurrent_requests(self) -> int:
        """Maximum concurrent requests across all servers."""
        return self._config.max_concurrent_per_server * len(self._servers)

    async def __aenter__(self) -> Self:
        """Initialize HTTP session and validate all servers.

        Validates that all servers are reachable and have consistent
        configuration (same model, same structure).
        """
        timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
        self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        # Validate all servers and get consistent configuration
        try:
            await self._validate_servers()
        except Exception:
            await self._session.close()
            raise

        logger.info(
            f"REST backend initialized with {len(self._servers)} servers, "
            f"max {self.max_concurrent_requests} concurrent requests"
        )
        if self._model_structure:
            logger.info(
                f"Model structure: {self._model_structure.num_layers} layers, "
                f"mixers: {self._model_structure.mixer_names}"
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close HTTP session."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Get the session, raising if not in context manager."""
        if self._session is None:
            raise RuntimeError(
                "RESTBackend must be used as an async context manager: "
                "async with RESTBackend(config) as backend: ..."
            )
        return self._session

    def _next_server(self) -> str:
        """Get next server URL using round-robin."""
        server = self._servers[self._idx % len(self._servers)]
        self._idx += 1
        return server

    async def _validate_servers(self) -> None:
        """Validate all servers are reachable and have consistent configuration.

        Fetches /info from all servers in parallel, verifies they're all healthy,
        and checks that they all have the same configuration.

        Raises:
            RESTBackendError: If any server is unreachable or configurations differ.
        """
        session = self._ensure_session()

        async def fetch_info(server: str) -> tuple[str, dict | None, str | None]:
            """Fetch info from a server, returning (server, data, error)."""
            try:
                async with session.get(f"{server}/info") as resp:
                    if resp.status == 200:
                        return (server, await resp.json(), None)
                    else:
                        return (server, None, f"HTTP {resp.status}")
            except Exception as e:
                return (server, None, str(e))

        results = await asyncio.gather(*[fetch_info(s) for s in self._servers])

        # Check for failures and collect successful results
        failed: list[tuple[str, str]] = []
        infos: list[tuple[str, dict]] = []
        for server, data, error in results:
            if error is not None:
                failed.append((server, error))
            else:
                assert data is not None  # If no error, data must exist
                infos.append((server, data))

        if failed:
            failed_msg = ", ".join(f"{s}: {e}" for s, e in failed)
            raise RESTBackendError(
                f"Failed to connect to {len(failed)}/{len(self._servers)} servers: {failed_msg}"
            )

        # All succeeded - validate consistency
        reference_server, reference_info = infos[0]

        # Check model_path consistency
        ref_model = reference_info.get("model_path")
        for server, info in infos[1:]:
            if info.get("model_path") != ref_model:
                raise RESTBackendError(
                    f"Configuration mismatch: {server} has model_path={info.get('model_path')}, "
                    f"but {reference_server} has model_path={ref_model}"
                )

        # Check supports_placement consistency
        ref_supports = reference_info.get("supports_placement")
        for server, info in infos[1:]:
            if info.get("supports_placement") != ref_supports:
                raise RESTBackendError(
                    f"Configuration mismatch: {server} has supports_placement="
                    f"{info.get('supports_placement')}, but {reference_server} "
                    f"has supports_placement={ref_supports}"
                )

        # If supernet, check num_layers and mixer_names consistency
        if ref_supports:
            ref_layers = reference_info.get("num_layers")
            ref_mixers = reference_info.get("mixer_names")

            # Validate reference server has required fields
            if ref_layers is None or ref_mixers is None:
                raise RESTBackendError(
                    f"Server {reference_server} reports supports_placement=True "
                    f"but missing num_layers or mixer_names"
                )

            for server, info in infos[1:]:
                if info.get("num_layers") != ref_layers:
                    raise RESTBackendError(
                        f"Configuration mismatch: {server} has num_layers="
                        f"{info.get('num_layers')}, but {reference_server} "
                        f"has num_layers={ref_layers}"
                    )
                if info.get("mixer_names") != ref_mixers:
                    raise RESTBackendError(
                        f"Configuration mismatch: {server} has mixer_names="
                        f"{info.get('mixer_names')}, but {reference_server} "
                        f"has mixer_names={ref_mixers}"
                    )

            # Store model structure
            self._model_structure = ModelStructure(
                num_layers=ref_layers,
                mixer_names=tuple(ref_mixers),
            )

        logger.info(
            f"All {len(self._servers)} servers validated: "
            f"model_path={ref_model}, supports_placement={ref_supports}"
        )

    @property
    def model_structure(self) -> ModelStructure:
        """Model structure (num_layers, mixer_names).

        Available after entering async context (__aenter__).

        Raises:
            NotImplementedError: If server doesn't expose model structure.
        """
        if self._model_structure is None:
            raise NotImplementedError(
                "Model structure not available from REST server. "
                "The server may be running a non-supernet model, or "
                "the server version doesn't support model structure discovery."
            )
        return self._model_structure

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
        """Generate completions via REST server.

        Args:
            messages: Chat messages for generation.
            n: Number of completions to generate.
            max_tokens: Maximum tokens per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            request_id: Optional request identifier.

        Returns:
            GenerationResult with samples and prompt token IDs.
        """
        from place_layers.evaluator.importance_weighted.vllm_client import (
            GenerationResult,
        )

        session = self._ensure_session()
        server = self._next_server()

        payload = {
            "messages": messages,
            "n": n,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "request_id": request_id,
            "placement": self._current_placement,
        }

        async with self._semaphore:
            async with session.post(f"{server}/generate", json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RESTBackendError(
                        f"Generate request failed (status {resp.status}): {error_text}"
                    )
                data = await resp.json()

        return GenerationResult.model_validate(data)

    async def compute_batch_log_probs(
        self,
        traces: Sequence[tuple[Sequence[int], Sequence[int]]],
    ) -> list[float]:
        """Compute log probabilities for traces, distributed across servers.

        Traces are distributed across servers in round-robin fashion.
        Each server processes its chunk and results are reassembled.

        Args:
            traces: List of (prompt_tokens, completion_tokens) tuples.

        Returns:
            List of log probabilities in input order.
        """
        if not traces:
            return []

        # Distribute traces across servers
        # Each chunk is (server_url, [(original_idx, trace), ...])
        chunks: list[tuple[str, list[tuple[int, tuple[Sequence[int], Sequence[int]]]]]] = [
            (server, []) for server in self._servers
        ]

        for idx, trace in enumerate(traces):
            server_idx = idx % len(self._servers)
            chunks[server_idx][1].append((idx, trace))

        # Process chunks in parallel
        tasks = []
        for server, chunk_traces in chunks:
            if chunk_traces:  # Skip empty chunks
                tasks.append(self._compute_chunk_log_probs(server, chunk_traces))

        chunk_results = await asyncio.gather(*tasks)

        # Reassemble results in original order
        results: list[tuple[int, float]] = []
        for chunk_result in chunk_results:
            results.extend(chunk_result)

        # Sort by original index and extract log probs
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    async def _compute_chunk_log_probs(
        self,
        server: str,
        indexed_traces: list[tuple[int, tuple[Sequence[int], Sequence[int]]]],
    ) -> list[tuple[int, float]]:
        """Compute log probs for a chunk of traces on a specific server.

        Args:
            server: Server URL.
            indexed_traces: List of (original_idx, (prompt, completion)) tuples.

        Returns:
            List of (original_idx, log_prob) tuples.
        """
        session = self._ensure_session()

        # Extract traces for request
        traces_payload = [
            {
                "prompt_tokens": list(trace[0]),
                "completion_tokens": list(trace[1]),
            }
            for _, trace in indexed_traces
        ]

        payload = {
            "traces": traces_payload,
            "placement": self._current_placement,
        }

        async with self._semaphore:
            async with session.post(
                f"{server}/compute_log_probs", json=payload
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RESTBackendError(
                        f"Compute log probs failed (status {resp.status}): {error_text}"
                    )
                data = await resp.json()

        log_probs = data["log_probs"]

        # Pair log probs with original indices
        return [(idx, lp) for (idx, _), lp in zip(indexed_traces, log_probs)]

    async def compute_text_log_probs(
        self,
        contexts: Sequence[str],
        continuations: Sequence[str],
    ) -> list[float]:
        """Compute text log-probs - NOT YET IMPLEMENTED for REST backend.

        The REST servers run TransformersBackend which supports text log-probs,
        but we need to add a /compute_text_log_probs endpoint to expose this.

        For now, use TransformersBackend directly for likelihood evaluation.
        """
        raise NotImplementedError(
            "Text-based log-prob scoring is not yet implemented for REST backend. "
            "The server needs a /compute_text_log_probs endpoint. "
            "For now, use TransformersBackend directly for likelihood evaluation."
        )

    async def compute_chat_log_probs(
        self,
        chats: Sequence[tuple[Sequence[ChatMessage], Sequence[ChatMessage]]],
    ) -> list[float]:
        """Compute chat log-probs - NOT YET IMPLEMENTED for REST backend.

        The REST servers run TransformersBackend which supports this,
        but we need to add a /compute_chat_log_probs endpoint to expose it.

        For now, use TransformersBackend directly for likelihood evaluation.
        """
        raise NotImplementedError(
            "Chat-based log-prob scoring is not yet implemented for REST backend. "
            "The server needs a /compute_chat_log_probs endpoint. "
            "For now, use TransformersBackend directly for likelihood evaluation."
        )

    async def set_placement(self, mixer_names: Sequence[str]) -> None:
        """Set mixer placement for subsequent requests.

        The placement is stored locally and included in every request.
        This allows load balancing without coordinating placement state
        across servers.

        Args:
            mixer_names: Mixer name for each layer.
        """
        self._current_placement = list(mixer_names)
        logger.debug(f"Placement set to: {self._current_placement}")


class RESTBackendConfig(BaseBackendConfig):
    """Configuration for REST backend."""

    _backend_class: ClassVar[type[AsyncBackend]] = RESTBackend

    backend_type: Literal["rest"] = "rest"

    servers: list[str] = Field(
        description="List of server URLs (e.g., ['http://localhost:8080', 'http://localhost:8081'])"
    )
    max_concurrent_per_server: int = Field(
        default=8,
        description="Maximum concurrent requests per server",
        ge=1,
    )
    timeout_seconds: int = Field(
        default=300,
        description="Request timeout in seconds",
        ge=1,
    )
