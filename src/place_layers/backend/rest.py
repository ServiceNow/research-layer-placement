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
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Self, Sequence

import aiohttp
from pydantic import BaseModel, Field, computed_field, model_validator
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_base

from place_layers.backend.base import AsyncBackend, BaseBackendConfig, concurrent_map
from place_layers.backend.types import (
    BackendError,
    GenerateChatRequest,
    GenerateTextRequest,
    LogProbResult,
    ModelStructure,
    NoStructure,
    ScoreChatRequest,
    ScoreChatResult,
    ScoreTextRequest,
    ScoreTextResult,
    ScoreTokensRequest,
    ScoreTokensResult,
    SupernetStructure,
)
from place_layers.clients.types import GenerationResult
from place_layers.server.schemas import (
    GenerateChatRequest as ServerGenerateChatRequest,
    GenerateTextRequest as ServerGenerateTextRequest,
    ScoreChatRequest as ServerScoreChatRequest,
    ScoreTextRequest as ServerScoreTextRequest,
    ScoreTokensRequest as ServerScoreTokensRequest,
)

logger = logging.getLogger(__name__)


class RESTBackendError(Exception):
    """Error communicating with REST server.

    Attributes:
        status_code: HTTP status code if available (4xx for client errors, 5xx for server errors).
        message: Error message.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message

    @property
    def error_type(self) -> Literal["bad_request", "timeout", "server_error"]:
        """Return error type based on status code."""
        if self.status_code is not None and 400 <= self.status_code < 500:
            return "bad_request"
        return "server_error"


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
    """Manages server health tracking and load-balanced server selection.

    This class has two responsibilities:

    1. **Health Tracking**: Tracks which servers are healthy and schedules
       recovery checks for unhealthy servers using exponential backoff.

    2. **Server Selection**: Provides load-balanced server selection via
       round-robin distribution across healthy servers.

    Thread Safety
    -------------
    The `select_server()` method is safe to call from concurrent async tasks.
    While `_round_robin_counter` is modified without locking, this is safe because:
    - Python's GIL ensures integer increment is atomic
    - There are no `await` points between read and write of the counter
    - Worst case of a "race" is slightly uneven distribution, not incorrectness

    Usage
    -----
        pool = ServerPool(["http://server1:8080", "http://server2:8080"], ...)

        # Select a server for load balancing
        preferred = pool.select_server()

        # Use with failover (tries preferred first, then others)
        for server in pool.get_servers_to_try():
            try:
                result = await call_server(server, ...)
                pool.mark_success(server)
                return result
            except TransientError:
                pool.mark_failure(server)
    """

    servers: list[str]
    base_recovery_interval: float
    max_recovery_interval: float
    _health: dict[str, ServerHealth] = field(default_factory=dict, repr=False)
    _round_robin_counter: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """Initialize health tracking for all servers."""
        self._health = {server: ServerHealth() for server in self.servers}
        self._round_robin_counter = 0

    # -------------------------------------------------------------------------
    # Server Selection (Load Balancing)
    # -------------------------------------------------------------------------

    def select_server(self) -> str | None:
        """Select the next server using round-robin load balancing.

        Distributes requests evenly across healthy servers. The round-robin
        counter operates on the full server list but skips unhealthy servers,
        ensuring consistent ordering even when servers fail and recover.

        Returns:
            The selected server URL, or None if no servers are available.

        Note:
            When None is returned, the caller should proceed with failover
            logic (via `get_servers_to_try()`) which will wait for server
            recovery if all servers are unhealthy.
        """
        available = self.get_servers_to_try()
        if not available:
            return None

        # Round-robin over the FULL server list, skipping unavailable servers.
        # This maintains stable ordering even when servers fail/recover.
        for _ in range(len(self.servers)):
            candidate = self.servers[self._round_robin_counter % len(self.servers)]
            self._round_robin_counter += 1
            if candidate in available:
                return candidate

        # Should never reach here if get_servers_to_try() returned non-empty
        return None

    def get_servers_to_try(self) -> list[str]:
        """Get all available servers in priority order for failover.

        Returns healthy servers first (preserving original order), followed by
        unhealthy servers that are due for a recovery check.

        This method is used by failover logic to determine which servers to
        try when a request fails. For initial server selection, use
        `select_server()` instead.
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

    # -------------------------------------------------------------------------
    # Health Tracking
    # -------------------------------------------------------------------------

    def mark_success(self, server: str) -> None:
        """Mark a server as healthy after a successful request."""
        if server in self._health:
            was_unhealthy = not self._health[server].healthy
            self._health[server].mark_healthy()
            if was_unhealthy:
                logger.info(f"Server {server} recovered and marked healthy")

    def mark_failure(self, server: str, reason: str = "unknown") -> None:
        """Mark a server as unhealthy after a failed request.

        Args:
            server: The server URL that failed.
            reason: Human-readable reason for the failure.
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
                f"Server {server} marked unhealthy: {reason} "
                f"(failures: {health.consecutive_failures}, "
                f"next check in {wait_time:.1f}s){reason_str}"
            )

    # -------------------------------------------------------------------------
    # Status Queries
    # -------------------------------------------------------------------------

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

    Servers run TransformersBackend via the serve command. Requests are
    distributed using round-robin load balancing with automatic failover.
    Each request includes the current placement, allowing servers to
    update their state lazily on receipt.
    """

    def __init__(self, config: "RESTBackendConfig | RESTDNSBackendConfig") -> None:
        self._config = config
        self._session: aiohttp.ClientSession | None = None
        # Per-server semaphores for concurrency control
        self._server_semaphores: dict[str, asyncio.Semaphore] | None = None
        self._current_placement: list[str] | None = None
        self._model_structure: SupernetStructure | None = None

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
            raise RuntimeError("RESTBackend must be used as an async context manager")
        return self._server_semaphores[server]

    async def _call_with_failover(
        self,
        endpoint: str,
        request: BaseModel,
        preferred_server: str | None = None,
    ) -> dict[str, Any]:
        """Call an endpoint with failover across servers.

        Tries preferred server first (if specified and healthy), then falls back
        to other healthy servers, then unhealthy servers due for recovery.
        Uses tenacity for retry with server-aware wait strategy.

        Args:
            endpoint: API endpoint (e.g., "/generate").
            request: Typed request model to serialize and send.
            preferred_server: Optional server to try first (for load distribution).

        Returns:
            JSON response from the server.

        Raises:
            RESTBackendError: If all servers fail after retries.
        """
        session = self._ensure_session()
        payload = request.model_dump(mode="json")

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
                servers_to_try = [preferred_server] + [
                    s for s in servers_to_try if s != preferred_server
                ]

            errors: list[tuple[str, str]] = []
            for server in servers_to_try:
                try:
                    async with self._get_semaphore(server):
                        async with session.post(
                            f"{server}{endpoint}", json=payload
                        ) as resp:
                            if resp.status == 503:
                                # Server unavailable - mark unhealthy, try next
                                reason = "503 Service Unavailable"
                                self._server_pool.mark_failure(server, reason)
                                errors.append((server, reason))
                                continue
                            if resp.status >= 500:
                                # Server error - mark unhealthy, try next
                                error_text = await resp.text()
                                reason = f"{resp.status}: {error_text[:200]}"
                                self._server_pool.mark_failure(server, reason)
                                errors.append((server, reason))
                                continue
                            if resp.status >= 400:
                                # Client error - don't retry, raise immediately
                                error_text = await resp.text()
                                raise RESTBackendError(
                                    f"Request failed (status {resp.status}): {error_text}",
                                    status_code=resp.status,
                                )
                            # Success - parse response (errors here are not connection issues)
                            self._server_pool.mark_success(server)
                            return await resp.json()
                except aiohttp.ClientConnectionError as e:
                    # Connection error - mark unhealthy, try next server
                    reason = f"Connection error: {e}"
                    self._server_pool.mark_failure(server, reason)
                    errors.append((server, reason))
                    continue
                except asyncio.TimeoutError:
                    # Timeout - mark unhealthy, try next server
                    reason = f"Timeout after {self._config.timeout_seconds}s"
                    self._server_pool.mark_failure(server, reason)
                    errors.append((server, reason))
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
            self._model_structure = SupernetStructure(
                num_layers=ref_layers,
                mixer_names=ref_mixers,
            )

        logger.info(
            f"All {len(self._server_pool.servers)} servers validated: "
            f"model_path={ref_model}, supports_placement={ref_supports}"
        )

    @property
    def model_structure(self) -> ModelStructure:
        """Model structure (num_layers, mixer_names).

        Available after entering async context (__aenter__).

        Returns SupernetStructure if the server has model structure info,
        NoStructure otherwise.
        """
        if self._model_structure is None:
            return NoStructure()
        return self._model_structure

    @property
    def placement(self) -> list[str] | None:
        """Current mixer placement, or None if not set."""
        return self._current_placement

    def generate_chat(
        self,
        requests: Iterable[GenerateChatRequest]
        | AsyncIterable[GenerateChatRequest | None],
    ) -> AsyncIterator[GenerationResult | BackendError]:
        """Generate completions from chat messages, yielding as they complete.

        Results may arrive in any order. Use key to join with input.
        Requests are distributed across healthy servers.

        On transient errors, yields BackendError instead of killing the stream.
        """

        async def process_one(
            req: GenerateChatRequest,
        ) -> GenerationResult | BackendError:
            try:
                # Get preferred server via round-robin for load distribution
                preferred_server = self._server_pool.select_server()
                server_request = ServerGenerateChatRequest(
                    key=req.key,
                    messages=req.messages,
                    n=req.n,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    placement=self._current_placement,
                )
                data = await self._call_with_failover(
                    "/generate_chat", server_request, preferred_server=preferred_server
                )
                return GenerationResult.model_validate(data)
            except RESTBackendError as e:
                logger.warning(f"REST error for {req.key}: {e}")
                return BackendError(key=req.key, error_type=e.error_type, message=str(e))

        return concurrent_map(requests, process_one, self.max_concurrent_requests)

    def generate_text(
        self,
        requests: Iterable[GenerateTextRequest]
        | AsyncIterable[GenerateTextRequest | None],
    ) -> AsyncIterator[GenerationResult | BackendError]:
        """Generate completions from text prompts, yielding as they complete.

        Results may arrive in any order. Use key to join with input.
        Requests are distributed across healthy servers.

        On transient errors, yields BackendError instead of killing the stream.
        """

        async def process_one(
            req: GenerateTextRequest,
        ) -> GenerationResult | BackendError:
            try:
                # Get preferred server via round-robin for load distribution
                preferred_server = self._server_pool.select_server()
                server_request = ServerGenerateTextRequest(
                    key=req.key,
                    prompt=req.prompt,
                    n=req.n,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    placement=self._current_placement,
                )
                data = await self._call_with_failover(
                    "/generate_text", server_request, preferred_server=preferred_server
                )
                return GenerationResult.model_validate(data)
            except RESTBackendError as e:
                logger.warning(f"REST error for {req.key}: {e}")
                return BackendError(key=req.key, error_type=e.error_type, message=str(e))

        return concurrent_map(requests, process_one, self.max_concurrent_requests)

    async def score_tokens(
        self,
        requests: Iterable[ScoreTokensRequest]
        | AsyncIterable[ScoreTokensRequest | None],
    ) -> AsyncIterator[ScoreTokensResult | BackendError]:
        """Score token sequences, yielding results as they complete.

        Requests are distributed across healthy servers.
        Results may arrive in any order.

        On transient errors, yields BackendError instead of killing the stream.
        """

        async def process_one(req: ScoreTokensRequest) -> ScoreTokensResult | BackendError:
            try:
                # Get preferred server via round-robin for load distribution
                preferred_server = self._server_pool.select_server()
                server_request = ServerScoreTokensRequest(
                    key=req.key,
                    prompt_tokens=req.prompt_tokens,
                    completion_tokens=req.completion_tokens,
                    placement=self._current_placement,
                )
                data = await self._call_with_failover(
                    "/score_tokens", server_request, preferred_server=preferred_server
                )
                return ScoreTokensResult(key=data["key"], log_prob=data["log_prob"])
            except RESTBackendError as e:
                logger.warning(f"REST error for {req.key}: {e}")
                return BackendError(key=req.key, error_type=e.error_type, message=str(e))

        async for result in concurrent_map(
            requests, process_one, self.max_concurrent_requests
        ):
            yield result

    async def score_text(
        self,
        requests: Iterable[ScoreTextRequest] | AsyncIterable[ScoreTextRequest | None],
    ) -> AsyncIterator[ScoreTextResult | BackendError]:
        """Score text continuations (multiple-choice), yielding as complete.

        Results may arrive in any order. Each result contains all choices
        for that request (diff-based scoring requires all choices together).

        On transient errors, yields BackendError instead of killing the stream.
        """

        async def process_one(req: ScoreTextRequest) -> ScoreTextResult | BackendError:
            try:
                if not req.choices:
                    return ScoreTextResult(key=req.key, results=[])

                # Get preferred server via round-robin for load distribution
                preferred_server = self._server_pool.select_server()
                server_request = ServerScoreTextRequest(
                    key=req.key,
                    context=req.context,
                    choices=req.choices,
                    placement=self._current_placement,
                )
                data = await self._call_with_failover(
                    "/score_text", server_request, preferred_server=preferred_server
                )

                results = [
                    LogProbResult(
                        token_ids=r["token_ids"],
                        logprobs=r["logprobs"],
                        answer_range=tuple(r["answer_range"]),
                        total_log_prob=r["total_log_prob"],
                    )
                    for r in data["results"]
                ]
                return ScoreTextResult(key=req.key, results=results)
            except RESTBackendError as e:
                logger.warning(f"REST error for {req.key}: {e}")
                return BackendError(key=req.key, error_type=e.error_type, message=str(e))

        async for result in concurrent_map(
            requests, process_one, self.max_concurrent_requests
        ):
            yield result

    async def score_chat(
        self,
        requests: Iterable[ScoreChatRequest] | AsyncIterable[ScoreChatRequest | None],
    ) -> AsyncIterator[ScoreChatResult | BackendError]:
        """Score chat completions (multiple-choice), yielding as complete.

        Results may arrive in any order. Each result contains all choices
        for that request (diff-based scoring requires all choices together).

        On transient errors, yields BackendError instead of killing the stream.
        """

        async def process_one(req: ScoreChatRequest) -> ScoreChatResult | BackendError:
            try:
                if not req.completion_messages:
                    return ScoreChatResult(key=req.key, results=[])

                # Get preferred server via round-robin for load distribution
                preferred_server = self._server_pool.select_server()
                server_request = ServerScoreChatRequest(
                    key=req.key,
                    context_messages=req.context_messages,
                    completion_messages=req.completion_messages,
                    placement=self._current_placement,
                )
                data = await self._call_with_failover(
                    "/score_chat", server_request, preferred_server=preferred_server
                )

                results = [
                    LogProbResult(
                        token_ids=r["token_ids"],
                        logprobs=r["logprobs"],
                        answer_range=tuple(r["answer_range"]),
                        total_log_prob=r["total_log_prob"],
                    )
                    for r in data["results"]
                ]
                return ScoreChatResult(key=req.key, results=results)
            except RESTBackendError as e:
                logger.warning(f"REST error for {req.key}: {e}")
                return BackendError(key=req.key, error_type=e.error_type, message=str(e))

        async for result in concurrent_map(
            requests, process_one, self.max_concurrent_requests
        ):
            yield result

    async def set_placement(self, placement: Sequence[str]) -> None:
        """Set mixer placement for subsequent requests.

        The placement is stored locally and sent with every request.
        Servers will update their state automatically when they receive
        a request with a different placement than their current state.

        This design ensures placement consistency even when servers
        recover from failures - the first request after recovery will
        update the server's placement.

        Args:
            placement: Mixer to use at each layer.
        """
        self._current_placement = list(placement)
        logger.debug(f"Placement set locally: {self._current_placement}")


class BaseRESTBackendConfig(BaseBackendConfig):
    """Base configuration for REST backends.

    Subclasses must provide the `servers` property (either as a field or computed).
    """

    _backend_class: ClassVar[type[AsyncBackend]] = RESTBackend

    max_concurrent_per_server: int = Field(
        description="Maximum concurrent requests per server",
        ge=1,
    )
    timeout_seconds: int = Field(
        description="Request timeout in seconds",
        ge=1,
    )
    base_recovery_interval: float = Field(
        description="Initial wait time before retrying an unhealthy server (seconds)",
        ge=0.1,
    )
    max_recovery_interval: float = Field(
        description="Maximum wait time between recovery checks (seconds)",
        ge=1.0,
    )
    max_retries: int = Field(
        description="Maximum number of retry attempts when all servers fail",
        ge=1,
    )


class RESTBackendConfig(BaseRESTBackendConfig):
    """REST backend with explicit server list."""

    backend_type: Literal["rest"] = "rest"

    servers: list[str] = Field(
        description="List of server URLs (e.g., ['http://localhost:8080', 'http://localhost:8081'])"
    )


class RESTDNSBackendConfig(BaseRESTBackendConfig):
    """REST backend with auto-generated server list from DNS pattern.

    Generates server URLs automatically based on a DNS prefix and either:
    1. A specific list of server IDs (via server_ids), or
    2. A sequential range of server IDs (via num_servers)

    Useful for Kubernetes deployments with predictable DNS naming schemes.

    Example with num_servers:
        dns_prefix = "my-service"
        num_servers = 4
        port = 8080

        Generates:
        - http://my-service-id-0:8080
        - http://my-service-id-1:8080
        - http://my-service-id-2:8080
        - http://my-service-id-3:8080

    Example with server_ids:
        dns_prefix = "my-service"
        server_ids = [0, 2, 5]
        port = 8080

        Generates:
        - http://my-service-id-0:8080
        - http://my-service-id-2:8080
        - http://my-service-id-5:8080
    """

    backend_type: Literal["rest_dns"] = "rest_dns"

    dns_prefix: str = Field(
        description="DNS prefix for server hostnames (servers named {dns_prefix}-id-{N})"
    )
    num_servers: int | None = Field(
        default=None,
        description="Number of servers to generate (creates sequential IDs 0..n-1). "
        "Mutually exclusive with server_ids.",
        ge=1,
    )
    server_ids: list[int] | None = Field(
        default=None,
        description="Explicit list of server IDs to use. Mutually exclusive with num_servers.",
    )
    port: int = Field(
        description="Port for all servers",
        ge=1,
        le=65535,
    )

    @model_validator(mode="after")
    def validate_server_specification(self) -> Self:
        """Validate that exactly one of num_servers or server_ids is provided."""
        has_num = self.num_servers is not None
        has_ids = self.server_ids is not None

        if not has_num and not has_ids:
            raise ValueError(
                "RESTDNSBackendConfig requires either 'num_servers' or 'server_ids' to be specified"
            )
        if has_num and has_ids:
            raise ValueError(
                "RESTDNSBackendConfig cannot have both 'num_servers' and 'server_ids' specified"
            )

        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def servers(self) -> list[str]:
        """Auto-generate server URLs from DNS prefix and server IDs."""
        if self.server_ids is not None:
            # Use explicit server IDs
            ids = self.server_ids
        elif self.num_servers is not None:
            # Use sequential range
            ids = list(range(self.num_servers))
        else:
            # Should never reach here due to __init__ validation
            raise ValueError("Either num_servers or server_ids must be specified")

        return [
            f"http://{self.dns_prefix}-id-{i}:{self.port}"
            for i in ids
        ]
