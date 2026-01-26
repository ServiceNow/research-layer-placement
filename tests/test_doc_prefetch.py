"""Integration tests for TaskBasedEvaluator doc prefetch mechanism and streaming.

These tests prove:
1. The async prefetch pattern works correctly (no deadlock)
2. Prefetch runs concurrently with evaluation
3. The streaming abstraction works at the base class level
4. Both async and sync backends work with the template method pattern

We use MockTaskEvaluator for prefetch tests and MockStreamingEvaluator
for testing the abstract hooks pattern.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal
from unittest.mock import MagicMock, patch

import pytest


@dataclass
class MockDoc:
    """Mock document."""
    task_name: str
    idx: int
    content: str


@dataclass
class MockTask:
    """Mock lm-eval task."""
    name: str
    output_type: str = "generate_until"

    @property
    def config(self):
        return self


class MockTaskEvaluator:
    """Mock evaluator that mirrors TaskBasedEvaluator's prefetch pattern.

    This simulates the exact async pattern used in production:
    - __aenter__ starts background prefetch task
    - _prefetch_docs loads docs with semaphore-limited concurrency
    - _prepare_task awaits asyncio.Event for each task's docs
    - Async generator yields requests while prefetch continues
    """

    def __init__(
        self,
        task_names: list[str],
        docs_per_task: int = 3,
        prefetch_delay: float = 0.05,
        prefetch_ahead: int = 2,
    ):
        self.task_names = task_names
        self.docs_per_task = docs_per_task
        self.prefetch_delay = prefetch_delay
        self.prefetch_ahead = prefetch_ahead

        # State (mirrors TaskBasedEvaluator)
        self._task_cache: dict[str, MockTask] = {}
        self._docs_cache: dict[str, list[MockDoc]] = {}
        self._docs_ready: dict[str, asyncio.Event] = {}
        self._prefetch_task: asyncio.Task | None = None

        # Timing tracking for tests
        self.prefetch_start_times: dict[str, float] = {}
        self.prefetch_complete_times: dict[str, float] = {}
        self.prepare_start_times: dict[str, float] = {}
        self.prepare_complete_times: dict[str, float] = {}
        self._start_time: float = 0

    async def __aenter__(self) -> "MockTaskEvaluator":
        """Enter context - mirrors TaskBasedEvaluator.__aenter__."""
        self._start_time = asyncio.get_event_loop().time()

        # Load tasks (simulated)
        self._task_cache = {name: MockTask(name) for name in self.task_names}

        # Initialize events
        for name in self._task_cache:
            self._docs_ready[name] = asyncio.Event()

        # Start background prefetch
        self._prefetch_task = asyncio.create_task(
            self._prefetch_docs(prefetch_ahead=self.prefetch_ahead)
        )

        return self

    async def __aexit__(self, *args):
        """Exit context."""
        if self._prefetch_task:
            await self._prefetch_task

    async def _prefetch_docs(self, prefetch_ahead: int) -> None:
        """Prefetch docs - mirrors TaskBasedEvaluator._prefetch_docs."""
        sem = asyncio.Semaphore(prefetch_ahead)

        async def load_one(task_name: str) -> None:
            async with sem:
                self.prefetch_start_times[task_name] = (
                    asyncio.get_event_loop().time() - self._start_time
                )

                # Simulate I/O delay
                await asyncio.sleep(self.prefetch_delay)

                # Populate cache
                self._docs_cache[task_name] = [
                    MockDoc(task_name, i, f"content_{i}")
                    for i in range(self.docs_per_task)
                ]

                self.prefetch_complete_times[task_name] = (
                    asyncio.get_event_loop().time() - self._start_time
                )

                # Signal ready
                self._docs_ready[task_name].set()

        await asyncio.gather(*[load_one(name) for name in self.task_names])

    async def _prepare_task(
        self, task_name: str
    ) -> tuple[MockTask, str, list[MockDoc]] | None:
        """Prepare task - mirrors TaskBasedEvaluator._prepare_task."""
        self.prepare_start_times[task_name] = (
            asyncio.get_event_loop().time() - self._start_time
        )

        # Wait for docs (this is the key - must not block event loop)
        await self._docs_ready[task_name].wait()

        self.prepare_complete_times[task_name] = (
            asyncio.get_event_loop().time() - self._start_time
        )

        task = self._task_cache[task_name]
        docs = self._docs_cache[task_name]
        return task, task_name, docs

    async def request_generator(self) -> AsyncIterator[dict]:
        """Async generator - mirrors _run_tasks_async pattern."""
        for task_name in self.task_names:
            prepared = await self._prepare_task(task_name)
            if prepared is None:
                continue

            task, name, docs = prepared
            for doc in docs:
                yield {"task": name, "doc": doc}


class TestNoDeadlock:
    """Tests proving the async pattern doesn't deadlock."""

    @pytest.mark.asyncio
    async def test_single_task_no_deadlock(self):
        """Single task: prefetch and prepare work without deadlock."""
        async with MockTaskEvaluator(["task_a"]) as evaluator:
            result = await evaluator._prepare_task("task_a")

        assert result is not None
        task, name, docs = result
        assert name == "task_a"
        assert len(docs) == 3

    @pytest.mark.asyncio
    async def test_multiple_tasks_no_deadlock(self):
        """Multiple tasks: all prepare without deadlock."""
        async with MockTaskEvaluator(["task_a", "task_b", "task_c"]) as evaluator:
            results = []
            for name in ["task_a", "task_b", "task_c"]:
                result = await evaluator._prepare_task(name)
                results.append(result)

        assert all(r is not None for r in results)
        assert [r[1] for r in results] == ["task_a", "task_b", "task_c"]

    @pytest.mark.asyncio
    async def test_async_generator_no_deadlock(self):
        """Async generator pattern works without deadlock."""
        async with MockTaskEvaluator(["task_a", "task_b"]) as evaluator:
            requests = [req async for req in evaluator.request_generator()]

        assert len(requests) == 6  # 2 tasks * 3 docs
        assert requests[0]["task"] == "task_a"
        assert requests[3]["task"] == "task_b"

    @pytest.mark.asyncio
    async def test_threading_event_would_deadlock(self):
        """Demonstrate that threading.Event would cause deadlock."""
        import threading

        docs_ready = threading.Event()

        async def prefetch():
            await asyncio.sleep(0.05)
            docs_ready.set()

        prefetch_task = asyncio.create_task(prefetch())

        # threading.Event.wait() blocks the event loop
        result = docs_ready.wait(timeout=0.2)

        prefetch_task.cancel()
        try:
            await prefetch_task
        except asyncio.CancelledError:
            pass

        assert result is False, "threading.Event.wait() should timeout (deadlock)"


class TestConcurrency:
    """Tests proving prefetch runs concurrently with evaluation."""

    @pytest.mark.asyncio
    async def test_prefetch_overlaps_with_prepare(self):
        """Prove prefetch of later tasks overlaps with prepare of earlier tasks.

        With prefetch_ahead=2 and 4 tasks:
        - task_a and task_b start prefetching immediately (concurrent)
        - While we prepare task_a, task_b prefetch continues
        - When task_a prepare completes, task_c starts prefetching
        """
        evaluator = MockTaskEvaluator(
            task_names=["task_a", "task_b", "task_c", "task_d"],
            prefetch_delay=0.05,
            prefetch_ahead=2,
        )

        async with evaluator:
            # Prepare tasks in order
            for name in ["task_a", "task_b", "task_c", "task_d"]:
                await evaluator._prepare_task(name)

        # task_a and task_b should start prefetching at nearly the same time
        # (both fit in semaphore=2)
        assert evaluator.prefetch_start_times["task_a"] < 0.01
        assert evaluator.prefetch_start_times["task_b"] < 0.01

        # task_c should start after one of the first two completes
        # (semaphore slot freed)
        assert evaluator.prefetch_start_times["task_c"] >= 0.04

        # Key assertion: task_b prefetch completes BEFORE we finish preparing task_a
        # This proves concurrency - prefetch runs while we wait
        assert evaluator.prefetch_complete_times["task_b"] <= evaluator.prepare_complete_times["task_a"] + 0.02

    @pytest.mark.asyncio
    async def test_later_tasks_prefetch_during_earlier_processing(self):
        """Prove later tasks are prefetched while processing earlier ones."""
        evaluator = MockTaskEvaluator(
            task_names=["task_a", "task_b", "task_c"],
            prefetch_delay=0.1,
            prefetch_ahead=3,  # All can prefetch concurrently
        )

        process_times: dict[str, float] = {}

        async with evaluator:
            async for req in evaluator.request_generator():
                # Simulate processing delay
                await asyncio.sleep(0.02)
                task_name = req["task"]
                process_times[task_name] = asyncio.get_event_loop().time() - evaluator._start_time

        # All tasks should complete prefetch around the same time
        # (concurrent with prefetch_ahead=3)
        prefetch_times = list(evaluator.prefetch_complete_times.values())
        assert max(prefetch_times) - min(prefetch_times) < 0.02, (
            "All prefetches should complete nearly simultaneously"
        )

        # task_c should be prefetched before task_a processing finishes
        assert evaluator.prefetch_complete_times["task_c"] < process_times["task_a"]


class TestSemaphore:
    """Tests for semaphore-based concurrency limiting."""

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrent_prefetch(self):
        """Verify semaphore limits concurrent prefetch operations."""
        current_concurrent = 0
        max_concurrent = 0
        original_sleep = asyncio.sleep

        async def tracking_sleep(delay):
            nonlocal current_concurrent, max_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            await original_sleep(delay)
            current_concurrent -= 1

        evaluator = MockTaskEvaluator(
            task_names=["t1", "t2", "t3", "t4", "t5", "t6"],
            prefetch_delay=0.05,
            prefetch_ahead=2,  # Only 2 concurrent
        )

        # Patch sleep to track concurrency
        import unittest.mock
        with unittest.mock.patch("asyncio.sleep", tracking_sleep):
            async with evaluator:
                pass  # Just run prefetch

        assert max_concurrent <= 2, f"Should not exceed 2 concurrent, got {max_concurrent}"
        assert max_concurrent == 2, f"Should reach 2 concurrent for efficiency"
        assert len(evaluator._docs_cache) == 6, "All tasks should be prefetched"

    @pytest.mark.asyncio
    async def test_prefetch_ahead_1_is_sequential(self):
        """With prefetch_ahead=1, tasks prefetch sequentially."""
        evaluator = MockTaskEvaluator(
            task_names=["task_a", "task_b", "task_c"],
            prefetch_delay=0.05,
            prefetch_ahead=1,  # Sequential
        )

        async with evaluator:
            pass

        # Each task should start after the previous completes
        assert evaluator.prefetch_start_times["task_b"] >= evaluator.prefetch_complete_times["task_a"] - 0.01
        assert evaluator.prefetch_start_times["task_c"] >= evaluator.prefetch_complete_times["task_b"] - 0.01


class TestRealEvaluatorIntegration:
    """Integration tests with real TaskBasedEvaluator components."""

    @pytest.mark.asyncio
    async def test_real_prefetch_docs_with_mock_tasks(self):
        """Test actual _prefetch_docs method with mocked task loading."""
        from unittest.mock import MagicMock, patch
        from place_layers.evaluator.task_based_evaluator import TaskBasedEvaluator

        # Create a concrete subclass for testing
        class TestableEvaluator(TaskBasedEvaluator):
            def __init__(self):
                # Skip parent __init__ which needs config
                self._task_cache = {}
                self._docs_cache = {}
                self._docs_ready = {}
                self._prefetch_task = None

            async def evaluate(self, parameters):
                pass

        evaluator = TestableEvaluator()

        # Set up mock tasks
        mock_task = MagicMock()
        mock_task.download = MagicMock()
        evaluator._task_cache = {"test_task": mock_task}
        evaluator._docs_ready = {"test_task": asyncio.Event()}

        # Mock config
        evaluator.config = MagicMock()
        evaluator.config.tasks = ["test_task"]
        evaluator.config.split = "test"

        # Mock get_docs
        with patch(
            "place_layers.evaluator.task_based_evaluator.get_docs",
            return_value=[{"q": "question", "a": "answer"}],
        ):
            await evaluator._prefetch_docs(prefetch_ahead=2)

        assert "test_task" in evaluator._docs_cache
        assert evaluator._docs_ready["test_task"].is_set()
        mock_task.download.assert_called_once()


# -----------------------------------------------------------------------------
# Streaming Abstraction Tests
# -----------------------------------------------------------------------------


@dataclass
class MockRequest:
    """Mock request for streaming tests."""

    key: str


@dataclass
class MockResult:
    """Mock result for streaming tests."""

    key: str
    data: str = "success"


class MockAsyncBackend:
    """Mock async backend for streaming tests."""

    def __init__(self, delay: float = 0.01):
        self.delay = delay
        self.requests_received: list[MockRequest] = []

    async def process(
        self, requests: AsyncIterable[MockRequest]
    ) -> AsyncIterator[MockResult]:
        """Process requests asynchronously."""
        async for req in requests:
            self.requests_received.append(req)
            await asyncio.sleep(self.delay)
            yield MockResult(key=req.key)


class MockSyncBackend:
    """Mock sync backend for streaming tests."""

    def __init__(self):
        self.requests_received: list[MockRequest] = []

    def process(self, requests: Iterable[MockRequest]) -> Iterator[MockResult]:
        """Process requests synchronously."""
        for req in requests:
            self.requests_received.append(req)
            yield MockResult(key=req.key)


class MockStreamingEvaluator:
    """Mock evaluator that implements the streaming hooks pattern.

    This tests the template method pattern at the TaskBasedEvaluator level,
    verifying that:
    1. _build_request is called for each problem
    2. _stream_to_backend_* routes correctly
    3. _process_result handles each result
    4. Cross-task streaming works correctly
    """

    def __init__(
        self,
        task_names: list[str],
        docs_per_task: int = 3,
        use_async: bool = True,
    ):
        self.task_names = task_names
        self.docs_per_task = docs_per_task
        self.use_async = use_async

        # State
        self._task_cache: dict[str, MockTask] = {}
        self._docs_cache: dict[str, list[MockDoc]] = {}
        self._docs_ready: dict[str, asyncio.Event] = {}
        self._prefetch_task: asyncio.Task | None = None

        # Mock backends
        self._async_backend = MockAsyncBackend()
        self._sync_backend = MockSyncBackend()

        # Tracking for verification
        self.build_request_calls: list[tuple[str, int]] = []
        self.process_result_calls: list[tuple[str, Any]] = []
        self.processed_count = 0

    @property
    def _backend(self):
        """Return the appropriate backend based on use_async."""
        return self._async_backend if self.use_async else self._sync_backend

    async def __aenter__(self) -> "MockStreamingEvaluator":
        """Enter context."""
        self._task_cache = {name: MockTask(name) for name in self.task_names}

        # Pre-populate docs cache (skip prefetch complexity for these tests)
        for name in self._task_cache:
            self._docs_cache[name] = [
                MockDoc(name, i, f"content_{i}")
                for i in range(self.docs_per_task)
            ]
            self._docs_ready[name] = asyncio.Event()
            self._docs_ready[name].set()

        return self

    async def __aexit__(self, *args):
        """Exit context."""
        pass

    # -------------------------------------------------------------------------
    # Abstract hook implementations (mirroring the real pattern)
    # -------------------------------------------------------------------------

    def _expected_output_type(self) -> Literal["generate_until", "multiple_choice"]:
        """Return expected output type."""
        return "generate_until"

    def _build_request(
        self, task: MockTask, doc: MockDoc, idx: int, task_name: str
    ) -> tuple[MockRequest, dict]:
        """Build request and context."""
        self.build_request_calls.append((task_name, idx))
        return MockRequest(key=f"{task_name}:{idx}"), {"doc": doc}

    def _stream_to_backend_async(
        self, requests: AsyncIterable[Any]
    ) -> AsyncIterator[Any]:
        """Stream to async backend."""
        return self._async_backend.process(requests)

    def _stream_to_backend_sync(self, requests: Iterable[Any]) -> Iterator[Any]:
        """Stream to sync backend."""
        return self._sync_backend.process(requests)

    def _process_result(
        self,
        result: MockResult,
        context: dict,
        task: MockTask,
        task_name: str,
        doc: MockDoc,
        idx: int,
    ) -> None:
        """Process result."""
        self.process_result_calls.append((result.key, context))
        self.processed_count += 1

    # -------------------------------------------------------------------------
    # Streaming methods (matching TaskBasedEvaluator pattern)
    # -------------------------------------------------------------------------

    async def _prepare_task(
        self, task_name: str
    ) -> tuple[MockTask, str, list[MockDoc], list[int]] | None:
        """Prepare task for evaluation."""
        await self._docs_ready[task_name].wait()
        task = self._task_cache[task_name]
        docs = self._docs_cache[task_name]
        remaining = list(range(len(docs)))
        return task, task_name, docs, remaining

    async def _run_streaming_async(self) -> int:
        """Run streaming evaluation with async backend."""
        key_to_context: dict[str, tuple[MockTask, str, MockDoc, int, dict]] = {}

        async def request_generator():
            for task_name in self.task_names:
                prepared = await self._prepare_task(task_name)
                if prepared is None:
                    continue

                task, name, docs, remaining = prepared

                for idx in remaining:
                    doc = docs[idx]
                    request, context = self._build_request(task, doc, idx, name)
                    key_to_context[request.key] = (task, name, doc, idx, context)
                    yield request

        results_stream = self._stream_to_backend_async(request_generator())
        processed = 0

        async for result in results_stream:
            task, task_name, doc, idx, context = key_to_context[result.key]
            self._process_result(result, context, task, task_name, doc, idx)
            processed += 1

        return processed

    async def _run_streaming_sync(self) -> int:
        """Run streaming evaluation with sync backend."""
        key_to_context: dict[str, tuple[MockTask, str, MockDoc, int, dict]] = {}
        prepared_tasks: list[tuple[MockTask, str, list[int]]] = []

        for task_name in self.task_names:
            prepared = await self._prepare_task(task_name)
            if prepared is None:
                continue

            task, name, docs, remaining = prepared
            prepared_tasks.append((task, name, remaining))

        def request_generator():
            for task, task_name, remaining in prepared_tasks:
                docs = self._docs_cache[task_name]
                for idx in remaining:
                    doc = docs[idx]
                    request, context = self._build_request(task, doc, idx, task_name)
                    key_to_context[request.key] = (task, task_name, doc, idx, context)
                    yield request

        results_iter = self._stream_to_backend_sync(request_generator())
        processed = 0

        for result in results_iter:
            task, task_name, doc, idx, context = key_to_context[result.key]
            self._process_result(result, context, task, task_name, doc, idx)
            processed += 1

        return processed


class TestStreamingAbstraction:
    """Tests for the streaming abstraction pattern."""

    @pytest.mark.asyncio
    async def test_async_streaming_processes_all_tasks(self):
        """Verify async streaming processes all tasks across multiple tasks."""
        async with MockStreamingEvaluator(
            ["task_a", "task_b"], docs_per_task=3, use_async=True
        ) as evaluator:
            processed = await evaluator._run_streaming_async()

        assert processed == 6  # 2 tasks * 3 docs
        assert evaluator.processed_count == 6

        # Verify all requests were built
        assert len(evaluator.build_request_calls) == 6
        assert ("task_a", 0) in evaluator.build_request_calls
        assert ("task_b", 2) in evaluator.build_request_calls

        # Verify all results were processed
        assert len(evaluator.process_result_calls) == 6

    @pytest.mark.asyncio
    async def test_sync_streaming_processes_all_tasks(self):
        """Verify sync streaming processes all tasks."""
        async with MockStreamingEvaluator(
            ["task_a", "task_b"], docs_per_task=3, use_async=False
        ) as evaluator:
            processed = await evaluator._run_streaming_sync()

        assert processed == 6
        assert evaluator.processed_count == 6
        assert len(evaluator.build_request_calls) == 6
        assert len(evaluator.process_result_calls) == 6

    @pytest.mark.asyncio
    async def test_cross_task_request_ordering(self):
        """Verify requests are built in task order."""
        async with MockStreamingEvaluator(
            ["alpha", "beta", "gamma"], docs_per_task=2, use_async=True
        ) as evaluator:
            await evaluator._run_streaming_async()

        # Requests should be built in order: alpha:0, alpha:1, beta:0, beta:1, gamma:0, gamma:1
        expected_order = [
            ("alpha", 0), ("alpha", 1),
            ("beta", 0), ("beta", 1),
            ("gamma", 0), ("gamma", 1),
        ]
        assert evaluator.build_request_calls == expected_order

    @pytest.mark.asyncio
    async def test_context_passed_to_process_result(self):
        """Verify context from _build_request is passed to _process_result."""
        async with MockStreamingEvaluator(
            ["task_a"], docs_per_task=2, use_async=True
        ) as evaluator:
            await evaluator._run_streaming_async()

        # Each result should have the context with the doc
        for key, context in evaluator.process_result_calls:
            assert "doc" in context
            assert isinstance(context["doc"], MockDoc)

    @pytest.mark.asyncio
    async def test_backend_receives_all_requests(self):
        """Verify backend receives all requests."""
        async with MockStreamingEvaluator(
            ["task_a", "task_b"], docs_per_task=3, use_async=True
        ) as evaluator:
            await evaluator._run_streaming_async()

        # Check async backend received all requests
        assert len(evaluator._async_backend.requests_received) == 6
        keys = [r.key for r in evaluator._async_backend.requests_received]
        assert "task_a:0" in keys
        assert "task_b:2" in keys

    @pytest.mark.asyncio
    async def test_sync_backend_receives_all_requests(self):
        """Verify sync backend receives all requests."""
        async with MockStreamingEvaluator(
            ["task_a", "task_b"], docs_per_task=3, use_async=False
        ) as evaluator:
            await evaluator._run_streaming_sync()

        # Check sync backend received all requests
        assert len(evaluator._sync_backend.requests_received) == 6
        keys = [r.key for r in evaluator._sync_backend.requests_received]
        assert "task_a:0" in keys
        assert "task_b:2" in keys

    @pytest.mark.asyncio
    async def test_empty_task_list_returns_zero(self):
        """Verify empty task list returns 0 processed."""
        async with MockStreamingEvaluator([], use_async=True) as evaluator:
            processed = await evaluator._run_streaming_async()

        assert processed == 0
        assert evaluator.processed_count == 0

    @pytest.mark.asyncio
    async def test_single_doc_per_task(self):
        """Verify single doc per task works correctly."""
        async with MockStreamingEvaluator(
            ["task_a", "task_b", "task_c"], docs_per_task=1, use_async=True
        ) as evaluator:
            processed = await evaluator._run_streaming_async()

        assert processed == 3
        assert evaluator.build_request_calls == [
            ("task_a", 0), ("task_b", 0), ("task_c", 0)
        ]


class TestStreamingInterleavedPrefetch:
    """Tests proving async streaming allows prefetch overlap."""

    @pytest.mark.asyncio
    async def test_async_streaming_with_delayed_prefetch(self):
        """Prove async streaming allows later task prefetch during earlier processing."""

        class DelayedPrefetchEvaluator(MockStreamingEvaluator):
            """Evaluator with delayed prefetch to prove interleaving."""

            def __init__(self, task_names, prefetch_delay=0.05):
                super().__init__(task_names, docs_per_task=2, use_async=True)
                self.prefetch_delay = prefetch_delay
                self.prefetch_times: dict[str, float] = {}
                self.process_times: dict[str, list[float]] = {n: [] for n in task_names}
                self._start_time = 0

            async def __aenter__(self):
                self._start_time = asyncio.get_event_loop().time()
                self._task_cache = {name: MockTask(name) for name in self.task_names}

                # Initialize events but don't set them yet
                for name in self._task_cache:
                    self._docs_ready[name] = asyncio.Event()

                # Start delayed prefetch
                asyncio.create_task(self._delayed_prefetch())
                return self

            async def _delayed_prefetch(self):
                """Prefetch tasks with delays to prove interleaving."""
                for name in self.task_names:
                    await asyncio.sleep(self.prefetch_delay)
                    self._docs_cache[name] = [
                        MockDoc(name, i, f"content_{i}")
                        for i in range(self.docs_per_task)
                    ]
                    self.prefetch_times[name] = (
                        asyncio.get_event_loop().time() - self._start_time
                    )
                    self._docs_ready[name].set()

            def _process_result(self, result, context, task, task_name, doc, idx):
                super()._process_result(result, context, task, task_name, doc, idx)
                self.process_times[task_name].append(
                    asyncio.get_event_loop().time() - self._start_time
                )

        async with DelayedPrefetchEvaluator(["task_a", "task_b", "task_c"]) as evaluator:
            await evaluator._run_streaming_async()

        # Verify all tasks processed
        assert evaluator.processed_count == 6

        # Verify prefetch completes for all tasks
        assert "task_a" in evaluator.prefetch_times
        assert "task_b" in evaluator.prefetch_times
        assert "task_c" in evaluator.prefetch_times

        # Verify prefetch happens in order (with delay between each)
        assert evaluator.prefetch_times["task_a"] < evaluator.prefetch_times["task_b"]
        assert evaluator.prefetch_times["task_b"] < evaluator.prefetch_times["task_c"]

        # Verify interleaving: task_a processing starts after task_a prefetch
        # but task_b/c prefetch can overlap with task_a processing
        if evaluator.process_times["task_a"]:
            first_task_a_process = min(evaluator.process_times["task_a"])
            # task_a processing should start after task_a prefetch completes
            assert first_task_a_process >= evaluator.prefetch_times["task_a"]
