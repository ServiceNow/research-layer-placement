"""REST backend for distributed TransformersBackend evaluation.

This module provides an AsyncBackend implementation that distributes
requests across multiple REST servers, each running a TransformersBackend.

Placement is managed via per-request placement fields:
- set_placement() stores placement locally on the client
- Every request includes the placement in its payload
- Servers update their state if the incoming placement differs
- This ensures consistency even when servers recover from failures
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal, Self, Sequence

import aiohttp
from pydantic import Field, model_validator
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_base

from place_layers.backend.base import (
    AsyncBackend,
    BaseBackendConfig,
    LogProbResult,
    ChatMessage,
    ModelStructure,
)

if TYPE_CHECKING:
    from place_layers.backend.vllm import GenerationResult

logger = logging.getLogger(__name__)


class RESTBackendError(Exception):
    """Error communicating with REST server."""

    pass


class TransientError(RESTBackendError):
    """Transient error that may be resolved by retrying on another server."""

    pass


class AllServersFailedError(TransientError):
    """All servers failed for a request (triggers retry with backoff)."""

    pass


@dataclass
class ServerHealth:
    """Health state for a single server with exponential recovery backoff."""

    healthy: bool = True
    consecutive_failures: int = 0
    last_failure_time: float = 0.0

    def mark_healthy(self) -> None:
        """Mark server as healthy, resetting failure state."""
        self.healthy = True
        self.consecutive_failures = 0

    def mark_unhealthy(self) -> None:
        """Mark server as unhealthy, incrementing failure count."""
        self.healthy = False
        self.consecutive_failures += 1
        self.last_failure_time = time.monotonic()

    def next_check_time(self, base_interval: float, max_interval: float) -> float:
        """Calculate when this server should next be checked for recovery."""
        if self.healthy:
            return 0.0  # Can be checked immediately
        wait = min(base_interval * (2 ** (self.consecutive_failures - 1)), max_interval)
        return self.last_failure_time + wait

    def is_due_for_check(self, base_interval: float, max_interval: float) -> bool:
        """Check if this server is due for a recovery check."""
        if self.healthy:
            return True
        return time.monotonic() >= self.next_check_time(base_interval, max_interval)


@dataclass
class ServerPool:
    """Manages a pool of servers with health tracking and recovery scheduling.

    Tracks which servers are healthy and schedules recovery checks for
    unhealthy servers using exponential backoff with a cap.

    Usage:
        pool = ServerPool(["http://server1:8080", "http://server2:8080"])

        # Get servers to try (healthy first, then unhealthy due for check)
        for server in pool.get_servers_to_try():
            try:
                result = await call_server(server, ...)
                pool.mark_success(server)
                return result
            except TransientError:
                pool.mark_failure(server)

        raise AllServersFailedError("No servers available")
    """

    servers: list[str]
    base_recovery_interval: float = 10.0  # Initial wait before retry
    max_recovery_interval: float = 300.0  # Cap at 5 minutes
    _health: dict[str, ServerHealth] = field(default_factory=dict, repr=False)
    _round_robin_counter: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """Initialize health tracking for all servers."""
        self._health = {server: ServerHealth() for server in self.servers}
        self._round_robin_counter = 0

    def get_servers_to_try(self) -> list[str]:
        """Get servers in priority order for trying a request.

        Returns healthy servers first (in original order), then unhealthy
        servers that are due for a recovery check.
        """
        healthy: list[str] = []
        due_for_check: list[str] = []

        for server in self.servers:
            health = self._health[server]
            if health.healthy:
                healthy.append(server)
            elif health.is_due_for_check(
                self.base_recovery_interval, self.max_recovery_interval
            ):
                due_for_check.append(server)

        return healthy + due_for_check

    def get_next_server_round_robin(self) -> str:
        """Get the next healthy server in round-robin order.

        Maintains a counter across calls to ensure even distribution across batches.
        The counter operates on the full server list, but skips unhealthy servers.
        This ensures consistent round-robin ordering even when servers fail.

        Returns:
            The selected server.

        Raises:
            ValueError: If no healthy servers are available.
        """
        servers_to_try = self.get_servers_to_try()
        if not servers_to_try:
            raise ValueError("No healthy servers available")

        # Round-robin over the FULL server list, but skip unhealthy ones
        # This maintains consistent ordering even when servers fail/recover
        attempts = 0
        while attempts < len(self.servers):
            candidate = self.servers[self._round_robin_counter % len(self.servers)]
            self._round_robin_counter += 1

            # Check if this server is in the healthy list
            if candidate in servers_to_try:
                return candidate

            attempts += 1

        # Should never reach here if get_servers_to_try() returned non-empty list
        raise ValueError("No healthy servers found in round-robin")

    def mark_success(self, server: str) -> None:
        """Mark a server as healthy after a successful request."""
        if server in self._health:
            was_unhealthy = not self._health[server].healthy
            self._health[server].mark_healthy()
            if was_unhealthy:
                logger.info(f"Server {server} recovered and marked healthy")

    def mark_failure(self, server: str, reason: str = "") -> None:
        """Mark a server as unhealthy after a failed request.

        Args:
            server: Server URL that failed.
            reason: Optional reason for the failure (for logging).
        """
        if server in self._health:
            self._health[server].mark_unhealthy()
            health = self._health[server]
            next_check = health.next_check_time(
                self.base_recovery_interval, self.max_recovery_interval
            )
            wait_time = next_check - time.monotonic()
            reason_str = f": {reason}" if reason else ""
            logger.warning(
                f"Server {server} marked unhealthy "
                f"(failures: {health.consecutive_failures}, "
                f"next check in {wait_time:.1f}s){reason_str}"
            )

    def next_recovery_check_time(self) -> float | None:
        """Get the earliest time when an unhealthy server should be rechecked.

        Returns None if all servers are healthy.
        """
        next_time: float | None = None
        for server in self.servers:
            health = self._health[server]
            if not health.healthy:
                check_time = health.next_check_time(
                    self.base_recovery_interval, self.max_recovery_interval
                )
                if next_time is None or check_time < next_time:
                    next_time = check_time
        return next_time

    def healthy_count(self) -> int:
        """Return the number of currently healthy servers."""
        return sum(1 for h in self._health.values() if h.healthy)

    def all_unhealthy(self) -> bool:
        """Check if all servers are currently unhealthy."""
        return self.healthy_count() == 0


class WaitForServerRecovery(wait_base):
    """Tenacity wait strategy that waits for server recovery checks.

    When all servers are unhealthy, waits until the next server is due
    for a recovery check rather than using a fixed backoff.
    """

    def __init__(
        self,
        server_pool: ServerPool,
        min_wait: float = 0.1,
        max_wait: float = 60.0,
    ) -> None:
        self.server_pool = server_pool
        self.min_wait = min_wait
        self.max_wait = max_wait

    def __call__(self, retry_state: RetryCallState) -> float:
        """Calculate wait time until next retry."""
        next_recovery = self.server_pool.next_recovery_check_time()
        if next_recovery is not None:
            # Wait until the next server is due for recovery check
            wait = max(self.min_wait, next_recovery - time.monotonic())
            return min(wait, self.max_wait)

        # All servers healthy but we're retrying? Shouldn't happen, but use min wait
        return self.min_wait


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
        self._session: aiohttp.ClientSession | None = None
        # Per-server semaphores for concurrency control
        self._server_semaphores: dict[str, asyncio.Semaphore] | None = None
        self._current_placement: list[str] | None = None
        self._model_structure: ModelStructure | None = None

        if not config.servers:
            raise ValueError("At least one server URL must be provided")

        # Server pool with health tracking
        self._server_pool = ServerPool(
            servers=list(config.servers),
            base_recovery_interval=config.base_recovery_interval,
            max_recovery_interval=config.max_recovery_interval,
        )

    @property
    def max_concurrent_requests(self) -> int:
        """Maximum concurrent requests across all servers."""
        return self._config.max_concurrent_per_server * len(self._server_pool.servers)

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
        # Create per-server semaphores for concurrency control
        self._server_semaphores = {
            server: asyncio.Semaphore(self._config.max_concurrent_per_server)
            for server in self._server_pool.servers
        }

        # Validate all servers and get consistent configuration
        try:
            await self._validate_servers()
        except Exception:
            await self._session.close()
            raise

        healthy = self._server_pool.healthy_count()
        total = len(self._server_pool.servers)
        logger.info(
            f"REST backend initialized with {healthy}/{total} healthy servers, "
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

    def _get_semaphore(self, server: str) -> asyncio.Semaphore:
        """Get the semaphore for a specific server."""
        if self._server_semaphores is None:
            raise RuntimeError(
                "RESTBackend must be used as an async context manager"
            )
        return self._server_semaphores[server]

    async def _call_with_failover(
        self,
        endpoint: str,
        payload: dict[str, Any],
        preferred_server: str | None = None,
    ) -> dict[str, Any]:
        """Call an endpoint with failover across servers.

        Tries preferred server first (if specified), then falls back to other
        healthy servers, then unhealthy servers due for recovery.
        Uses tenacity for retry with server-aware wait strategy.

        Args:
            endpoint: API endpoint (e.g., "/generate").
            payload: JSON payload to send.
            preferred_server: Optional server to try first before others.

        Returns:
            JSON response from the server.

        Raises:
            RESTBackendError: If all servers fail after retries.
        """
        @retry(
            retry=retry_if_exception_type(AllServersFailedError),
            wait=WaitForServerRecovery(self._server_pool),
            stop=stop_after_attempt(self._config.max_retries),
            reraise=True,
        )
        async def _with_retry() -> dict[str, Any]:
            servers_to_try = self._server_pool.get_servers_to_try()
            if not servers_to_try:
                # No servers available - wait for recovery
                raise AllServersFailedError(
                    "No servers available to try (all unhealthy, none due for check)"
                )

            # If preferred server is specified and healthy, try it first
            if preferred_server is not None and preferred_server in servers_to_try:
                # Move preferred server to front of list
                servers_to_try = [preferred_server] + [
                    s for s in servers_to_try if s != preferred_server
                ]

            errors: list[tuple[str, str]] = []
            for server in servers_to_try:
                try:
                    # Use common HTTP request logic
                    return await self._do_http_request(server, endpoint, payload)
                except RESTBackendError as e:
                    # Server failed - try next server
                    logger.debug(f"Server {server} failed for {endpoint}: {e}")
                    errors.append((server, str(e)))
                    continue

            # All servers we tried failed
            error_msg = "; ".join(f"{s}: {e}" for s, e in errors)
            raise AllServersFailedError(f"All available servers failed: {error_msg}")

        try:
            return await _with_retry()
        except AllServersFailedError as e:
            raise RESTBackendError(str(e)) from e

    async def _do_http_request(
        self,
        server: str,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute an HTTP POST request to a specific server.

        Handles response status codes, marks server health, and logs the request.
        This is the core HTTP logic shared by both _call_server and _call_with_failover.

        Args:
            server: Server URL to call.
            endpoint: API endpoint (e.g., "/compute_chat_log_probs").
            payload: JSON payload to send.

        Returns:
            JSON response from the server.

        Raises:
            RESTBackendError: If the request fails (4xx, 5xx, or connection error).
        """
        session = self._ensure_session()

        try:
            async with self._get_semaphore(server):
                async with session.post(f"{server}{endpoint}", json=payload) as resp:
                    if resp.status == 503:
                        error_msg = "503 Service Unavailable"
                        self._server_pool.mark_failure(server, error_msg)
                        raise RESTBackendError(f"{server} returned {error_msg}")
                    if resp.status >= 500:
                        error_text = await resp.text()
                        error_msg = f"{resp.status}: {error_text}"
                        self._server_pool.mark_failure(server, error_msg)
                        raise RESTBackendError(
                            f"{server} returned {error_msg}"
                        )
                    if resp.status >= 400:
                        error_text = await resp.text()
                        raise RESTBackendError(
                            f"Request failed (status {resp.status}): {error_text}"
                        )
                    # Success
                    self._server_pool.mark_success(server)
                    logger.debug(f"Request to {endpoint} handled by {server}")
                    return await resp.json()
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
            error_msg = f"Connection error: {type(e).__name__}: {e}"
            self._server_pool.mark_failure(server, error_msg)
            raise RESTBackendError(f"{error_msg} to {server}") from e

    async def _validate_servers(self) -> None:
        """Validate servers and check configuration consistency.

        Fetches /info from all servers in parallel. Servers that fail to respond
        are marked as unhealthy and will be retried during evaluation. At least
        one server must be reachable. All reachable servers must have consistent
        configuration.

        Raises:
            RESTBackendError: If no servers are reachable or configurations differ.
        """
        session = self._ensure_session()

        async def fetch_info(server: str) -> tuple[str, dict | None, str | None]:
            """Fetch info from a server, returning (server, data, error)."""
            try:
                # Use a short timeout for validation to fail fast on unhealthy servers
                timeout = aiohttp.ClientTimeout(total=10.0)
                async with session.get(f"{server}/info", timeout=timeout) as resp:
                    if resp.status == 200:
                        return (server, await resp.json(), None)
                    else:
                        return (server, None, f"HTTP {resp.status}")
            except Exception as e:
                return (server, None, str(e))

        results = await asyncio.gather(
            *[fetch_info(s) for s in self._server_pool.servers]
        )

        # Check for failures and collect successful results
        failed: list[tuple[str, str]] = []
        infos: list[tuple[str, dict]] = []
        for server, data, error in results:
            if error is not None:
                failed.append((server, error))
                # Mark as unhealthy - will be retried during evaluation
                self._server_pool.mark_failure(server, error)
            else:
                assert data is not None  # If no error, data must exist
                infos.append((server, data))

        if not infos:
            # No servers reachable at all
            failed_msg = ", ".join(f"{s}: {e}" for s, e in failed)
            raise RESTBackendError(
                f"Failed to connect to all {len(self._server_pool.servers)} servers: {failed_msg}"
            )

        if failed:
            # Some servers failed but at least one succeeded
            failed_msg = ", ".join(f"{s}: {e}" for s, e in failed)
            logger.warning(
                f"Failed to connect to {len(failed)}/{len(self._server_pool.servers)} servers "
                f"during initialization: {failed_msg}. These servers will be retried during evaluation."
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
            f"All {len(self._server_pool.servers)} servers validated: "
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
        from place_layers.backend.vllm import (
            GenerationResult,
        )

        payload = {
            "messages": messages,
            "n": n,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "request_id": request_id,
            "placement": self._current_placement,
        }

        # Use round-robin to select the preferred server for load balancing
        preferred_server = self._server_pool.get_next_server_round_robin()
        data = await self._call_with_failover("/generate", payload, preferred_server)
        return GenerationResult.model_validate(data)

    def _parse_log_prob_result(self, r: dict[str, Any]) -> "LogProbResult":
        """Parse a server response dict into a LogProbResult.

        Args:
            r: Server response dict with token_ids, logprobs, answer_range, total_log_prob.

        Returns:
            LogProbResult object.
        """
        return LogProbResult(
            token_ids=tuple(r["token_ids"]),
            logprobs=tuple(r["logprobs"]),
            answer_range=tuple(r["answer_range"]),
            total_log_prob=r["total_log_prob"],
        )

    def _extract_indexed_results[R](
        self, indexed_results: list[tuple[int, R]]
    ) -> list[R]:
        """Sort indexed results by index and extract values.

        Args:
            indexed_results: List of (index, result) tuples.

        Returns:
            List of results sorted by index.
        """
        indexed_results.sort(key=lambda x: x[0])
        return [r[1] for r in indexed_results]

    def _get_available_servers(self, context: str = "") -> list[str]:
        """Get available servers or raise if none available.

        Args:
            context: Optional context string for error message (e.g., "batch processing").

        Returns:
            List of available servers.

        Raises:
            RESTBackendError: If no servers are available.
        """
        servers = self._server_pool.get_servers_to_try()
        if not servers:
            msg = "No servers available"
            if context:
                msg += f" for {context}"
            raise RESTBackendError(msg)
        return servers

    async def compute_batch_log_probs(
        self,
        traces: Sequence[tuple[Sequence[int], Sequence[int]]],
    ) -> list[float]:
        """Compute log probabilities for traces, distributed across servers.

        Traces are distributed across healthy servers. Each server processes
        its chunk and results are reassembled.

        Args:
            traces: List of (prompt_tokens, completion_tokens) tuples.

        Returns:
            List of log probabilities in input order.
        """
        if not traces:
            return []

        # Get servers to use (healthy + due for recovery check)
        servers = self._get_available_servers("batch processing")

        # Distribute traces across servers
        indexed_traces: list[tuple[int, tuple[Sequence[int], Sequence[int]]]] = [
            (idx, trace) for idx, trace in enumerate(traces)
        ]

        results = await self._distribute_batch(
            items=indexed_traces,
            endpoint="/compute_log_probs",
            make_payload=lambda items: {
                "traces": [
                    {
                        "prompt_tokens": list(trace[0]),
                        "completion_tokens": list(trace[1]),
                    }
                    for _, trace in items
                ],
                "placement": self._current_placement,
            },
            parse_response=lambda items, data: [
                (idx, lp) for (idx, _), lp in zip(items, data["log_probs"])
            ],
        )

        # Sort by original index and extract log probs
        return self._extract_indexed_results(results)

    async def _distribute_batch[T, R](
        self,
        items: list[tuple[int, T]],
        endpoint: str,
        make_payload: Callable[[list[tuple[int, T]]], dict[str, Any]],
        parse_response: Callable[[list[tuple[int, T]], dict[str, Any]], list[tuple[int, R]]],
    ) -> list[tuple[int, R]]:
        """Distribute batch items across servers with failover.

        Args:
            items: List of (index, item) tuples to process.
            endpoint: API endpoint to call.
            make_payload: Function to create request payload from items.
            parse_response: Function to parse response into (index, result) tuples.

        Returns:
            List of (index, result) tuples.
        """
        servers = self._get_available_servers()

        # Distribute items across servers using round-robin
        # Counter operates on full server list to maintain consistent ordering
        # even when servers fail/recover
        server_items: dict[str, list[tuple[int, Any]]] = {s: [] for s in servers}
        for item in items:
            server = self._server_pool.get_next_server_round_robin()
            server_items[server].append(item)

        # Process each server's chunk
        async def process_chunk(
            server: str, chunk_items: list[tuple[int, Any]]
        ) -> list[tuple[int, Any]]:
            if not chunk_items:
                return []
            logger.debug(f"_distribute_batch: Assigned {len(chunk_items)} items to {server}")
            payload = make_payload(chunk_items)
            # Call with failover, preferring the assigned server
            data = await self._call_with_failover(endpoint, payload, preferred_server=server)
            return parse_response(chunk_items, data)

        tasks = [
            process_chunk(server, chunk_items)
            for server, chunk_items in server_items.items()
            if chunk_items
        ]

        chunk_results = await asyncio.gather(*tasks)

        # Flatten results
        results: list[tuple[int, Any]] = []
        for chunk in chunk_results:
            results.extend(chunk)
        return results

    async def compute_text_log_probs(
        self,
        text_groups: Sequence[tuple[str, Sequence[str]]],
    ) -> list[LogProbResult]:
        """Compute log probabilities for text continuations with diff-based scoring.

        Distributes text groups across healthy servers for parallel computation.
        Each server applies diff-based scoring internally.

        Args:
            text_groups: List of (context, choices) tuples.

        Returns:
            List of LogProbResult (flattened across all groups) with diff-based
            answer_range and total_log_prob.
        """
        if not text_groups:
            return []

        # Create indexed items: (group_idx, (context, choices))
        # We track group index and choice count to reassemble results
        items: list[tuple[int, tuple[str, Sequence[str]]]] = [
            (idx, (context, choices))
            for idx, (context, choices) in enumerate(text_groups)
        ]

        # Capture current placement for the closure
        placement = self._current_placement

        def make_payload(
            chunk_items: list[tuple[int, tuple[str, Sequence[str]]]]
        ) -> dict[str, Any]:
            return {
                "text_groups": [
                    (context, list(choices))
                    for _, (context, choices) in chunk_items
                ],
                "placement": placement,
            }

        def parse_response(
            chunk_items: list[tuple[int, tuple[str, Sequence[str]]]],
            data: dict[str, Any],
        ) -> list[tuple[int, LogProbResult]]:
            # Server returns flattened results, we need to tag each with group index
            results: list[tuple[int, LogProbResult]] = []
            result_idx = 0
            for group_idx, (_, choices) in chunk_items:
                for _ in choices:
                    r = data["results"][result_idx]
                    results.append((group_idx, self._parse_log_prob_result(r)))
                    result_idx += 1
            return results

        indexed_results = await self._distribute_batch(
            items=items,
            endpoint="/compute_text_log_probs",
            make_payload=make_payload,
            parse_response=parse_response,
        )

        # Sort by group index then flatten (results already in choice order within groups)
        return self._extract_indexed_results(indexed_results)

    async def compute_chat_log_probs(
        self,
        chats: Sequence[tuple[Sequence[ChatMessage], Sequence[ChatMessage]]],
    ) -> list[LogProbResult]:
        """Compute log probabilities for chat completions with diff-based scoring.

        Groups chats by context before distributing to servers. This ensures that
        all choices for the same question go to the same server, which is required
        for diff-based scoring to work correctly.

        Like compute_text_log_probs, this method:
        1. Groups chats by shared context (same question = same context)
        2. Distributes groups (not individual chats) across servers
        3. Each server receives all choices for its assigned questions
        4. Server-side diff-based scoring identifies answer token ranges

        Args:
            chats: List of (context_messages, completion_messages) tuples.

        Returns:
            List of LogProbResult with token IDs, per-token logprobs,
            answer range, and total log probability (one per input chat, in order).
        """
        if not chats:
            return []

        from collections import defaultdict

        # Group chats by context (same context = same multiple-choice question)
        # Maps context_key -> list of (original_idx, completion_messages)
        context_groups: dict[tuple, list[tuple[int, Sequence[ChatMessage]]]] = (
            defaultdict(list)
        )
        context_msgs_by_key: dict[tuple, Sequence[ChatMessage]] = {}

        for i, (ctx_msgs, comp_msgs) in enumerate(chats):
            # Use message content as key (same question → same key)
            ctx_key = tuple((m["role"], m["content"]) for m in ctx_msgs)
            context_groups[ctx_key].append((i, comp_msgs))
            context_msgs_by_key[ctx_key] = ctx_msgs

        # Create items as groups: (group_idx, (context_msgs, list[completion_msgs]))
        # Track mapping from group_idx back to original chat indices
        group_to_original_indices: dict[int, list[int]] = {}
        items: list[tuple[int, tuple[Sequence[ChatMessage], list[Sequence[ChatMessage]]]]] = []

        for group_idx, (ctx_key, group) in enumerate(context_groups.items()):
            ctx_msgs = context_msgs_by_key[ctx_key]
            completion_msgs_list = [comp_msgs for _, comp_msgs in group]
            original_indices = [orig_idx for orig_idx, _ in group]

            items.append((group_idx, (ctx_msgs, completion_msgs_list)))
            group_to_original_indices[group_idx] = original_indices

        # Capture current placement for the closure
        placement = self._current_placement

        def make_payload(
            chunk_items: list[tuple[int, tuple[Sequence[ChatMessage], list[Sequence[ChatMessage]]]]]
        ) -> dict[str, Any]:
            # Convert groups to server-side format
            # Server expects: chats = [(context_msgs, completion_msgs), ...]
            # We need to flatten groups back into individual chats for the server
            all_chats: list[tuple[list[ChatMessage], list[ChatMessage]]] = []
            for _, (context, completions) in chunk_items:
                for completion in completions:
                    all_chats.append((list(context), list(completion)))

            return {
                "chats": all_chats,
                "placement": placement,
            }

        def parse_response(
            chunk_items: list[tuple[int, tuple[Sequence[ChatMessage], list[Sequence[ChatMessage]]]]],
            data: dict[str, Any],
        ) -> list[tuple[int, LogProbResult]]:
            # Server returns flattened results (one per chat)
            # We need to tag each result with its original index
            results: list[tuple[int, LogProbResult]] = []
            result_idx = 0

            for group_idx, (_, completions) in chunk_items:
                original_indices = group_to_original_indices[group_idx]
                for orig_idx in original_indices:
                    r = data["results"][result_idx]
                    results.append((orig_idx, self._parse_log_prob_result(r)))
                    result_idx += 1

            return results

        indexed_results = await self._distribute_batch(
            items=items,
            endpoint="/compute_chat_log_probs",
            make_payload=make_payload,
            parse_response=parse_response,
        )

        # Sort by original index and extract results
        return self._extract_indexed_results(indexed_results)

    async def set_placement(self, mixer_names: Sequence[str]) -> None:
        """Set mixer placement for subsequent requests.

        The placement is stored locally and sent with every request.
        Servers will update their state automatically when they receive
        a request with a different placement than their current state.

        This design ensures placement consistency even when servers
        recover from failures - the first request after recovery will
        update the server's placement.

        Args:
            mixer_names: Mixer name for each layer.
        """
        self._current_placement = list(mixer_names)
        logger.debug(f"Placement set locally: {self._current_placement}")


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
    base_recovery_interval: float = Field(
        default=10.0,
        description="Initial wait time before retrying an unhealthy server (seconds)",
        ge=0.1,
    )
    max_recovery_interval: float = Field(
        default=300.0,
        description="Maximum wait time between recovery checks (seconds)",
        ge=1.0,
    )
    max_retries: int = Field(
        default=10,
        description="Maximum number of retry attempts when all servers fail",
        ge=1,
    )


class RESTDNSBackendConfig(RESTBackendConfig):
    """Configuration for REST backend with automatic DNS-based server list generation.

    Generates server URLs automatically based on a DNS prefix and server count.
    Useful for distributed deployments with predictable DNS naming schemes.

    Example:
        dns_prefix = "dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f"
        num_servers = 4
        port = 8080

        Generates:
        [
            "http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-0:8080",
            "http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-1:8080",
            "http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-2:8080",
            "http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-3:8080"
        ]
    """

    backend_type: Literal["rest_dns"] = "rest_dns"

    dns_prefix: str = Field(
        description="DNS prefix for server hostnames (servers named {dns_prefix}-id-{0..num_servers-1})"
    )
    num_servers: int = Field(
        description="Number of servers to generate",
        ge=1,
    )
    port: int = Field(
        default=8080,
        description="Port for all servers (uniform across servers)",
        ge=1,
        le=65535,
    )

    # Override servers to make it optional with a default
    servers: list[str] = Field(
        default_factory=list,
        description="Auto-generated server URLs (do not set manually)",
    )

    @model_validator(mode="before")
    @classmethod
    def generate_servers(cls, data: dict) -> dict:
        """Auto-generate server list from DNS prefix and server count."""
        # Only generate if servers not already provided
        if not data.get("servers"):
            dns_prefix = data.get("dns_prefix")
            num_servers = data.get("num_servers")
            port = data.get("port", 8080)

            if dns_prefix is not None and num_servers is not None:
                data["servers"] = [
                    f"http://{dns_prefix}-id-{i}:{port}"
                    for i in range(int(num_servers))
                ]
        return data
