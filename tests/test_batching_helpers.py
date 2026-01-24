"""Tests for batching helper functions.

Tests cover:
- batched_map: Simple 1:1 batching
- batched_expand_map: Fan-out/batch/fan-in with variable expansion sizes
- batched_map_by_key: Batching with flush on key change
- concurrent_map: Async concurrent processing with bounded concurrency
- Timeout/heartbeat functionality for streaming mode

IMPORTANT: All tests use generators (not lists) as input to ensure the helpers
work correctly with single-pass iterators, as they would in production with
streaming bridges.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any

import pytest

from place_layers.backend.base import (
    batched_expand_map,
    batched_map,
    batched_map_by_key,
    concurrent_expand_map,
    concurrent_map,
)


def sync_stream[T](items: list[T]) -> Iterator[T]:
    """Convert a list to a single-pass sync generator.

    This ensures tests verify that helpers work with iterators that can only
    be consumed once, matching production behavior with streaming bridges.
    """
    yield from items


async def async_stream[T](items: list[T], heartbeat_every: int = 0) -> AsyncIterator[T | None]:
    """Convert a list to an async generator.

    Args:
        items: Items to yield.
        heartbeat_every: If > 0, yield None heartbeat every N items.
    """
    for i, item in enumerate(items):
        if heartbeat_every > 0 and i > 0 and i % heartbeat_every == 0:
            yield None  # Heartbeat
        yield item


# =============================================================================
# batched_map tests
# =============================================================================


class TestBatchedMap:
    """Tests for batched_map - simple 1:1 batching."""

    def test_basic_batching(self):
        """Items are grouped into batches of the specified size."""
        batches_seen: list[list[int]] = []

        def process(batch: list[int]) -> list[int]:
            batches_seen.append(list(batch))
            return [x * 2 for x in batch]

        result = list(batched_map(sync_stream(list(range(10))), process, batch_size=3))

        assert batches_seen == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        assert result == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

    def test_exact_batch_size_multiple(self):
        """When input length is exact multiple of batch_size, no partial batch."""
        batches_seen: list[list[int]] = []

        def process(batch: list[int]) -> list[int]:
            batches_seen.append(list(batch))
            return batch

        result = list(batched_map(sync_stream(list(range(6))), process, batch_size=2))

        assert batches_seen == [[0, 1], [2, 3], [4, 5]]
        assert result == [0, 1, 2, 3, 4, 5]

    def test_empty_input(self):
        """Empty input yields no results and calls process zero times."""
        call_count = 0

        def process(batch: list[int]) -> list[int]:
            nonlocal call_count
            call_count += 1
            return batch

        result = list(batched_map(sync_stream([]), process, batch_size=3))

        assert result == []
        assert call_count == 0

    def test_single_item(self):
        """Single item creates one partial batch."""
        batches_seen: list[list[int]] = []

        def process(batch: list[int]) -> list[int]:
            batches_seen.append(list(batch))
            return [x + 100 for x in batch]

        result = list(batched_map(sync_stream([42]), process, batch_size=10))

        assert batches_seen == [[42]]
        assert result == [142]

    def test_batch_size_larger_than_input(self):
        """When batch_size > len(items), all items go in one batch."""
        batches_seen: list[list[int]] = []

        def process(batch: list[int]) -> list[int]:
            batches_seen.append(list(batch))
            return batch

        result = list(batched_map(sync_stream([1, 2, 3]), process, batch_size=100))

        assert batches_seen == [[1, 2, 3]]
        assert result == [1, 2, 3]

    def test_batch_size_one(self):
        """batch_size=1 processes items individually."""
        batches_seen: list[list[int]] = []

        def process(batch: list[int]) -> list[int]:
            batches_seen.append(list(batch))
            return [x + 1 for x in batch]

        result = list(batched_map(sync_stream([10, 20, 30]), process, batch_size=1))

        assert batches_seen == [[10], [20], [30]]
        assert result == [11, 21, 31]

    def test_preserves_order(self):
        """Results maintain input order."""

        def process(batch: list[str]) -> list[str]:
            return [s.upper() for s in batch]

        result = list(batched_map(sync_stream(["a", "b", "c", "d", "e"]), process, batch_size=2))

        assert result == ["A", "B", "C", "D", "E"]

    def test_process_can_return_generator(self):
        """Process function can return any iterable, not just list."""

        def process(batch: list[int]) -> Iterable[int]:
            for x in batch:
                yield x * 2

        result = list(batched_map(sync_stream([1, 2, 3, 4]), process, batch_size=2))

        assert result == [2, 4, 6, 8]


# =============================================================================
# batched_map_by_key tests
# =============================================================================


class TestBatchedMapByKey:
    """Tests for batched_map_by_key - batching with flush on key change."""

    def test_same_key_batches_together(self):
        """Items with same key accumulate into batches."""
        batches_seen: list[tuple[str, list[tuple[str, int]]]] = []

        def get_key(item: tuple[str, int]) -> str:
            return item[0]

        def process(key: str, batch: list[tuple[str, int]]) -> list[int]:
            batches_seen.append((key, list(batch)))
            return [x[1] * 2 for x in batch]

        items = [("a", 1), ("a", 2), ("a", 3), ("a", 4), ("a", 5)]
        result = list(batched_map_by_key(sync_stream(items), get_key, process, batch_size=3))

        # Two batches: [1,2,3] and [4,5]
        assert batches_seen == [
            ("a", [("a", 1), ("a", 2), ("a", 3)]),
            ("a", [("a", 4), ("a", 5)]),
        ]
        assert result == [2, 4, 6, 8, 10]

    def test_key_change_triggers_flush(self):
        """When key changes, current batch is flushed before new key starts."""
        batches_seen: list[tuple[str, list[tuple[str, int]]]] = []

        def get_key(item: tuple[str, int]) -> str:
            return item[0]

        def process(key: str, batch: list[tuple[str, int]]) -> list[int]:
            batches_seen.append((key, list(batch)))
            return [x[1] for x in batch]

        items = [("a", 1), ("a", 2), ("b", 3), ("b", 4)]
        result = list(batched_map_by_key(sync_stream(items), get_key, process, batch_size=10))

        # Key change forces flush even though batch not full
        assert batches_seen == [
            ("a", [("a", 1), ("a", 2)]),
            ("b", [("b", 3), ("b", 4)]),
        ]
        assert result == [1, 2, 3, 4]

    def test_alternating_keys(self):
        """Each key change triggers a flush."""
        batches_seen: list[tuple[str, list[tuple[str, int]]]] = []

        def get_key(item: tuple[str, int]) -> str:
            return item[0]

        def process(key: str, batch: list[tuple[str, int]]) -> list[int]:
            batches_seen.append((key, list(batch)))
            return [x[1] for x in batch]

        items = [("a", 1), ("b", 2), ("a", 3), ("b", 4)]
        result = list(batched_map_by_key(sync_stream(items), get_key, process, batch_size=10))

        # Each item triggers a flush due to key change
        assert batches_seen == [
            ("a", [("a", 1)]),
            ("b", [("b", 2)]),
            ("a", [("a", 3)]),
            ("b", [("b", 4)]),
        ]
        assert result == [1, 2, 3, 4]

    def test_batch_size_triggers_flush_before_key_change(self):
        """Batch size limit triggers flush even with same key."""
        batches_seen: list[tuple[str, list[tuple[str, int]]]] = []

        def get_key(item: tuple[str, int]) -> str:
            return item[0]

        def process(key: str, batch: list[tuple[str, int]]) -> list[int]:
            batches_seen.append((key, list(batch)))
            return [x[1] for x in batch]

        items = [("a", 1), ("a", 2), ("a", 3), ("a", 4), ("b", 5)]
        result = list(batched_map_by_key(sync_stream(items), get_key, process, batch_size=2))

        # batch_size=2 flushes [1,2], [3,4], then key change, then [5]
        assert batches_seen == [
            ("a", [("a", 1), ("a", 2)]),
            ("a", [("a", 3), ("a", 4)]),
            ("b", [("b", 5)]),
        ]
        assert result == [1, 2, 3, 4, 5]

    def test_empty_input(self):
        """Empty input yields no results."""
        call_count = 0

        def get_key(item: Any) -> str:
            return "key"

        def process(key: str, batch: list[Any]) -> list[Any]:
            nonlocal call_count
            call_count += 1
            return batch

        result = list(batched_map_by_key(sync_stream([]), get_key, process, batch_size=10))

        assert result == []
        assert call_count == 0

    def test_single_item(self):
        """Single item creates one batch."""
        batches_seen: list[tuple[str, list[int]]] = []

        def get_key(item: int) -> str:
            return "k"

        def process(key: str, batch: list[int]) -> list[int]:
            batches_seen.append((key, list(batch)))
            return [x * 10 for x in batch]

        result = list(batched_map_by_key(sync_stream([42]), get_key, process, batch_size=10))

        assert batches_seen == [("k", [42])]
        assert result == [420]

    def test_tuple_key(self):
        """Key can be any hashable type (e.g., tuple for generation params)."""
        # Simulates generation requests with (n, max_tokens, temp, top_p) as key
        batches_seen: list[tuple[tuple, list]] = []

        def get_key(item: tuple) -> tuple:
            return item[0]

        def process(key: tuple, batch: list) -> list[str]:
            batches_seen.append((key, [x[1] for x in batch]))
            return [x[1] for x in batch]

        items = [
            ((1, 100, 0.0, 1.0), "req1"),
            ((1, 100, 0.0, 1.0), "req2"),  # same params
            ((2, 200, 0.7, 0.9), "req3"),  # different params
            ((2, 200, 0.7, 0.9), "req4"),  # same as req3
        ]
        result = list(batched_map_by_key(sync_stream(items), get_key, process, batch_size=10))

        assert batches_seen == [
            ((1, 100, 0.0, 1.0), ["req1", "req2"]),
            ((2, 200, 0.7, 0.9), ["req3", "req4"]),
        ]
        assert result == ["req1", "req2", "req3", "req4"]

    def test_preserves_order(self):
        """Results maintain input order within and across batches."""

        def get_key(item: tuple[str, int]) -> str:
            return item[0]

        def process(key: str, batch: list[tuple[str, int]]) -> list[int]:
            return [x[1] for x in batch]

        items = [(k, i) for i, k in enumerate(["a", "a", "b", "b", "a", "a"])]
        result = list(batched_map_by_key(sync_stream(items), get_key, process, batch_size=10))

        assert result == [0, 1, 2, 3, 4, 5]

    def test_heartbeat_with_timeout_flush(self):
        """None heartbeats trigger timeout-based flush when timeout elapsed."""

        def gen_items():
            yield ("a", 1)
            yield ("a", 2)
            time.sleep(0.02)  # Let timeout elapse
            yield None  # heartbeat - should trigger flush
            yield ("a", 3)

        batches_seen: list[list[int]] = []

        def get_key(item: tuple[str, int]) -> str:
            return item[0]

        def process(key: str, batch: list[tuple[str, int]]) -> list[int]:
            batches_seen.append([x[1] for x in batch])
            return [x[1] for x in batch]

        # flush_timeout=0.01 - heartbeat after 0.02s sleep should trigger flush
        result = list(
            batched_map_by_key(
                gen_items(), get_key, process, batch_size=10, flush_timeout=0.01
            )
        )

        # Heartbeat triggers flush after timeout, then final batch
        assert batches_seen == [[1, 2], [3]]
        assert result == [1, 2, 3]

    def test_heartbeat_no_flush_if_timeout_not_elapsed(self):
        """Heartbeat doesn't flush if timeout hasn't elapsed."""
        batches_seen: list[list[int]] = []

        def get_key(item: tuple[str, int]) -> str:
            return item[0]

        def process(key: str, batch: list[tuple[str, int]]) -> list[int]:
            batches_seen.append([x[1] for x in batch])
            return [x[1] for x in batch]

        # Items are consumed quickly - timeout won't elapse
        def gen_items():
            yield ("a", 1)
            yield None  # heartbeat - but timeout not elapsed
            yield ("a", 2)

        # Long timeout - heartbeat won't trigger flush since items arrive quickly
        result = list(
            batched_map_by_key(
                gen_items(), get_key, process, batch_size=10, flush_timeout=1000
            )
        )

        # No early flush, both items in same batch
        assert batches_seen == [[1, 2]]
        assert result == [1, 2]


# =============================================================================
# batched_expand_map tests
# =============================================================================


class TestBatchedExpandMap:
    """Tests for batched_expand_map - fan-out/batch/fan-in pattern."""

    def test_basic_expansion(self):
        """Basic case: each input expands to multiple elements."""
        batches_seen: list[list[str]] = []

        def expand(inp: tuple[str, list[str]]) -> tuple[str, list[str]]:
            return inp

        def process_batch(elements: list[str]) -> list[str]:
            batches_seen.append(list(elements))
            return [e.upper() for e in elements]

        def combine(key: str, results: list[str]) -> tuple[str, tuple[str, ...]]:
            return (key, tuple(results))

        inputs = [("req0", ["a", "b"]), ("req1", ["c", "d", "e"])]
        result = list(
            batched_expand_map(sync_stream(inputs), expand, process_batch, combine, batch_size=4)
        )

        assert batches_seen == [["a", "b", "c", "d"], ["e"]]
        assert result == [("req0", ("A", "B")), ("req1", ("C", "D", "E"))]

    def test_empty_expansion_preserves_order(self):
        """Empty expansions are yielded in correct order."""

        def expand(inp: tuple[str, list[str]]) -> tuple[str, list[str]]:
            return inp

        def process_batch(elements: list[str]) -> list[str]:
            return [e.upper() for e in elements]

        def combine(key: str, results: list[str]) -> tuple[str, int]:
            return (key, len(results))

        inputs = [("req0", ["a", "b"]), ("req1", []), ("req2", ["c"])]
        result = list(
            batched_expand_map(sync_stream(inputs), expand, process_batch, combine, batch_size=10)
        )

        # Order must be preserved: req0, req1, req2
        assert result == [("req0", 2), ("req1", 0), ("req2", 1)]

    def test_single_request_spans_multiple_batches(self):
        """One input with many elements spans multiple batches."""
        batches_seen: list[list[int]] = []

        def expand(inp: tuple[str, list[int]]) -> tuple[str, list[int]]:
            return inp

        def process_batch(elements: list[int]) -> list[int]:
            batches_seen.append(list(elements))
            return [e * 2 for e in elements]

        def combine(key: str, results: list[int]) -> tuple[str, list[int]]:
            return (key, results)

        inputs = [("req0", list(range(10)))]
        result = list(
            batched_expand_map(sync_stream(inputs), expand, process_batch, combine, batch_size=4)
        )

        # Elements split across batches
        assert batches_seen == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
        # But combined back into single result
        assert result == [("req0", [0, 2, 4, 6, 8, 10, 12, 14, 16, 18])]

    def test_mixed_sizes_with_empty(self):
        """Mix of empty, small, and large expansions."""
        batches_seen: list[list[str]] = []

        def expand(inp: tuple[str, list[str]]) -> tuple[str, list[str]]:
            return inp

        def process_batch(elements: list[str]) -> list[str]:
            batches_seen.append(list(elements))
            return elements

        def combine(key: str, results: list[str]) -> tuple[str, int]:
            return (key, len(results))

        inputs = [
            ("a", ["x", "y"]),  # 2 elements
            ("b", []),  # empty
            ("c", ["1", "2", "3", "4", "5", "6"]),  # 6 elements (spans 2 batches)
            ("d", ["z"]),  # 1 element
        ]
        result = list(
            batched_expand_map(sync_stream(inputs), expand, process_batch, combine, batch_size=4)
        )

        # Batches: [x,y,1,2], [3,4,5,6], [z]
        assert batches_seen == [["x", "y", "1", "2"], ["3", "4", "5", "6"], ["z"]]
        # Results in order with correct counts
        assert result == [("a", 2), ("b", 0), ("c", 6), ("d", 1)]

    def test_none_return_skips_item(self):
        """When expand returns None, item is skipped entirely."""

        def expand(inp: tuple[str, list[int]]) -> tuple[str, list[int]] | None:
            key, elements = inp
            if key == "skip":
                return None
            return (key, elements)

        def process_batch(elements: list[int]) -> list[int]:
            return elements

        def combine(key: str, results: list[int]) -> str:
            return key

        inputs = [("keep1", [1]), ("skip", [2]), ("keep2", [3])]
        result = list(
            batched_expand_map(sync_stream(inputs), expand, process_batch, combine, batch_size=10)
        )

        assert result == ["keep1", "keep2"]

    def test_all_empty_expansions(self):
        """All inputs have empty expansions."""

        def expand(inp: tuple[str, list[Any]]) -> tuple[str, list[Any]]:
            return inp

        def process_batch(elements: list[Any]) -> list[Any]:
            raise AssertionError("Should not be called for empty expansions")

        def combine(key: str, results: list[Any]) -> str:
            assert results == []
            return key

        inputs = [("a", []), ("b", []), ("c", [])]
        result = list(
            batched_expand_map(sync_stream(inputs), expand, process_batch, combine, batch_size=10)
        )

        assert result == ["a", "b", "c"]

    def test_empty_input(self):
        """Empty input yields no results."""
        call_count = 0

        def expand(inp: Any) -> tuple[str, list[Any]]:
            nonlocal call_count
            call_count += 1
            return ("key", [])

        def process_batch(elements: list[Any]) -> list[Any]:
            return elements

        def combine(key: str, results: list[Any]) -> str:
            return key

        result = list(
            batched_expand_map(sync_stream([]), expand, process_batch, combine, batch_size=10)
        )

        assert result == []
        assert call_count == 0

    def test_exact_batch_boundaries(self):
        """Elements align exactly with batch boundaries."""
        batches_seen: list[list[int]] = []

        def expand(inp: tuple[str, list[int]]) -> tuple[str, list[int]]:
            return inp

        def process_batch(elements: list[int]) -> list[int]:
            batches_seen.append(list(elements))
            return elements

        def combine(key: str, results: list[int]) -> tuple[str, list[int]]:
            return (key, results)

        inputs = [("a", [1, 2, 3, 4]), ("b", [5, 6, 7, 8])]
        result = list(
            batched_expand_map(sync_stream(inputs), expand, process_batch, combine, batch_size=4)
        )

        assert batches_seen == [[1, 2, 3, 4], [5, 6, 7, 8]]
        assert result == [("a", [1, 2, 3, 4]), ("b", [5, 6, 7, 8])]

    def test_result_order_within_expansion(self):
        """Results within an expansion maintain element order."""
        batches_seen: list[list[int]] = []

        def expand(inp: tuple[str, list[int]]) -> tuple[str, list[int]]:
            return inp

        def process_batch(elements: list[int]) -> list[int]:
            batches_seen.append(list(elements))
            # Return in same order
            return [e * 10 for e in elements]

        def combine(key: str, results: list[int]) -> list[int]:
            return results

        inputs = [("req", list(range(7)))]
        result = list(
            batched_expand_map(sync_stream(inputs), expand, process_batch, combine, batch_size=3)
        )

        assert batches_seen == [[0, 1, 2], [3, 4, 5], [6]]
        # Results recombined in original order
        assert result == [[0, 10, 20, 30, 40, 50, 60]]


# =============================================================================
# concurrent_map tests
# =============================================================================


class TestSyncAsyncDispatch:
    """Tests verifying sync/async dispatch works correctly in concurrent helpers."""

    def test_sync_stream_has_no_aiter(self):
        """Sync streams don't have __aiter__, so dispatch goes to sync path."""
        s = sync_stream([1, 2, 3])
        assert not hasattr(s, "__aiter__")
        assert hasattr(s, "__iter__")

    def test_async_stream_has_aiter(self):
        """Async streams have __aiter__, so dispatch goes to async path."""
        s = async_stream([1, 2, 3])
        assert hasattr(s, "__aiter__")
        # Async iterators don't have __iter__
        assert not hasattr(s, "__iter__")

    @pytest.mark.asyncio
    async def test_async_path_handles_heartbeats(self):
        """Async path correctly skips None heartbeats."""

        async def process(x: int) -> int:
            return x * 2

        # Use async_stream with heartbeats
        results: list[int] = []
        async for result in concurrent_map(
            async_stream([1, 2, 3], heartbeat_every=1), process, max_concurrent=2
        ):
            results.append(result)

        # Heartbeats (None values) should be skipped, only real items processed
        assert results == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_sync_and_async_produce_same_results(self):
        """Sync and async paths produce identical results for same input."""

        async def process(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * 2

        items = [1, 2, 3, 4, 5]

        # Run with sync stream
        sync_results: list[int] = []
        async for result in concurrent_map(
            sync_stream(items), process, max_concurrent=2
        ):
            sync_results.append(result)

        # Run with async stream
        async_results: list[int] = []
        async for result in concurrent_map(
            async_stream(items), process, max_concurrent=2
        ):
            async_results.append(result)

        # Both should produce identical results
        assert sync_results == async_results == [2, 4, 6, 8, 10]


@pytest.mark.parametrize("stream", [sync_stream, async_stream])
class TestConcurrentMap:
    """Tests for concurrent_map - async concurrent processing.

    All tests are parametrized to run with both sync and async iterables,
    ensuring both code paths produce identical results.
    """

    @pytest.mark.asyncio
    async def test_basic_concurrent(self, stream):
        """Basic concurrent processing of items."""

        async def process(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        results: list[int] = []
        async for result in concurrent_map(stream([1, 2, 3, 4, 5]), process, max_concurrent=3):
            results.append(result)

        # Results should be in input order
        assert results == [2, 4, 6, 8, 10]

    @pytest.mark.asyncio
    async def test_respects_max_concurrent(self, stream):
        """Never exceeds max_concurrent active tasks."""
        concurrent_count = 0
        max_observed = 0

        async def process(x: int) -> int:
            nonlocal concurrent_count, max_observed
            concurrent_count += 1
            max_observed = max(max_observed, concurrent_count)
            await asyncio.sleep(0.02)
            concurrent_count -= 1
            return x

        results: list[int] = []
        async for result in concurrent_map(stream(list(range(10))), process, max_concurrent=3):
            results.append(result)

        assert max_observed <= 3
        assert results == list(range(10))

    @pytest.mark.asyncio
    async def test_empty_input(self, stream):
        """Empty input yields no results."""
        call_count = 0

        async def process(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x

        results: list[int] = []
        async for result in concurrent_map(stream([]), process, max_concurrent=5):
            results.append(result)

        assert results == []
        assert call_count == 0

    @pytest.mark.asyncio
    async def test_single_item(self, stream):
        """Single item is processed correctly."""

        async def process(x: int) -> int:
            return x * 10

        results: list[int] = []
        async for result in concurrent_map(stream([42]), process, max_concurrent=5):
            results.append(result)

        assert results == [420]

    @pytest.mark.asyncio
    async def test_max_concurrent_one(self, stream):
        """max_concurrent=1 processes items sequentially."""
        execution_order: list[int] = []

        async def process(x: int) -> int:
            execution_order.append(x)
            await asyncio.sleep(0.01)
            return x

        results: list[int] = []
        async for result in concurrent_map(stream([1, 2, 3, 4, 5]), process, max_concurrent=1):
            results.append(result)

        # With max_concurrent=1, should process in order
        assert execution_order == [1, 2, 3, 4, 5]
        assert results == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_max_concurrent_exceeds_items(self, stream):
        """max_concurrent > len(items) processes all concurrently."""
        start_times: dict[int, float] = {}
        loop = asyncio.get_event_loop()

        async def process(x: int) -> int:
            start_times[x] = loop.time()
            await asyncio.sleep(0.05)
            return x

        results: list[int] = []
        async for result in concurrent_map(stream([1, 2, 3]), process, max_concurrent=10):
            results.append(result)

        # All items should start at nearly the same time
        times = list(start_times.values())
        assert max(times) - min(times) < 0.02  # All started within 20ms
        assert sorted(results) == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_exception_propagates(self, stream):
        """Exceptions from process function propagate correctly."""

        async def process(x: int) -> int:
            if x == 3:
                raise ValueError("Bad value")
            return x

        with pytest.raises(ValueError, match="Bad value"):
            async for _ in concurrent_map(stream([1, 2, 3, 4]), process, max_concurrent=2):
                pass

    @pytest.mark.asyncio
    async def test_results_yielded_in_input_order(self, stream):
        """Results are yielded in input order, not completion order."""
        completion_order: list[int] = []

        async def process(x: int) -> int:
            # Items complete in reverse order (5 finishes first, 1 finishes last)
            await asyncio.sleep((5 - x) * 0.01)
            completion_order.append(x)
            return x

        results: list[int] = []
        async for result in concurrent_map(stream([1, 2, 3, 4, 5]), process, max_concurrent=5):
            results.append(result)

        # Even though items complete in reverse order (5, 4, 3, 2, 1),
        # results are yielded in input order
        assert results == [1, 2, 3, 4, 5]
        # But internally they completed in reverse order
        assert completion_order == [5, 4, 3, 2, 1]


# =============================================================================
# batched_map timeout tests
# =============================================================================


class TestBatchedMapTimeout:
    """Tests for timeout/heartbeat functionality in batched_map."""

    def test_heartbeat_triggers_flush_on_timeout(self):
        """Heartbeat (None) triggers flush when timeout has elapsed."""

        def gen_items():
            yield 1
            yield 2
            time.sleep(0.02)  # Let timeout elapse
            yield None  # heartbeat - should trigger flush
            yield 3

        batches_seen: list[list[int]] = []

        def process(batch: list[int]) -> list[int]:
            batches_seen.append(list(batch))
            return [x * 10 for x in batch]

        result = list(batched_map(gen_items(), process, batch_size=10, flush_timeout=0.01))

        # Heartbeat should flush [1,2] after timeout, then [3] at end
        assert batches_seen == [[1, 2], [3]]
        assert result == [10, 20, 30]

    def test_heartbeat_no_flush_if_timeout_not_elapsed(self):
        """Heartbeat doesn't cause flush when timeout hasn't elapsed."""
        items: list[int | None] = [1, None, 2]  # heartbeat between items
        batches_seen: list[list[int]] = []

        def process(batch: list[int]) -> list[int]:
            batches_seen.append(list(batch))
            return batch

        # Very long timeout - heartbeat won't trigger flush
        result = list(batched_map(items, process, batch_size=10, flush_timeout=1000))

        # All items in one batch
        assert batches_seen == [[1, 2]]
        assert result == [1, 2]

    def test_multiple_heartbeats_only_flush_once(self):
        """Multiple consecutive heartbeats only flush once (if timeout elapsed)."""

        def gen_items():
            yield 1
            yield 2
            time.sleep(0.02)
            yield None  # First heartbeat - triggers flush
            yield None  # Second heartbeat - nothing to flush
            yield None  # Third heartbeat - nothing to flush
            yield 3

        batches_seen: list[list[int]] = []

        def process(batch: list[int]) -> list[int]:
            batches_seen.append(list(batch))
            return batch

        result = list(batched_map(gen_items(), process, batch_size=10, flush_timeout=0.01))

        # First heartbeat flushes [1,2], subsequent heartbeats do nothing (empty)
        assert batches_seen == [[1, 2], [3]]
        assert result == [1, 2, 3]

    def test_heartbeat_with_no_pending_items(self):
        """Heartbeat with empty pending list does nothing."""
        items: list[int | None] = [None, None, 1, 2]  # heartbeats before items
        batches_seen: list[list[int]] = []

        def process(batch: list[int]) -> list[int]:
            batches_seen.append(list(batch))
            return batch

        # Even with timeout, heartbeats with empty pending do nothing
        result = list(batched_map(items, process, batch_size=10, flush_timeout=0.001))

        # Just one batch at the end
        assert batches_seen == [[1, 2]]
        assert result == [1, 2]


# =============================================================================
# batched_expand_map timeout tests
# =============================================================================


class TestBatchedExpandMapTimeout:
    """Tests for timeout/heartbeat functionality in batched_expand_map."""

    def test_heartbeat_flushes_partial_batch(self):
        """Heartbeat with timeout flushes partial element batch when elapsed."""

        def gen_items():
            yield ("a", [1, 2, 3])  # 3 elements
            time.sleep(0.02)  # Let timeout elapse
            yield None  # heartbeat - should flush
            yield ("b", [4, 5])  # 2 elements

        batches_seen: list[list[int]] = []

        def expand(inp: tuple[str, list[int]]) -> tuple[str, list[int]]:
            return inp

        def process_batch(elements: list[int]) -> list[int]:
            batches_seen.append(list(elements))
            return [e * 10 for e in elements]

        def combine(key: str, results: list[int]) -> tuple[str, list[int]]:
            return (key, results)

        result = list(
            batched_expand_map(
                gen_items(), expand, process_batch, combine, batch_size=10, flush_timeout=0.01
            )
        )

        # Heartbeat flushes first 3 elements after timeout, then final 2
        assert batches_seen == [[1, 2, 3], [4, 5]]
        assert result == [("a", [10, 20, 30]), ("b", [40, 50])]

    def test_heartbeat_no_flush_when_batch_empty(self):
        """Heartbeat with empty pending doesn't cause issues."""
        items: list[tuple[str, list[int]] | None] = [
            None,  # heartbeat with nothing pending
            ("a", [1, 2]),
            None,  # heartbeat after complete item
        ]
        batches_seen: list[list[int]] = []

        def expand(inp: tuple[str, list[int]]) -> tuple[str, list[int]]:
            return inp

        def process_batch(elements: list[int]) -> list[int]:
            batches_seen.append(list(elements))
            return elements

        def combine(key: str, results: list[int]) -> str:
            return key

        result = list(
            batched_expand_map(
                items, expand, process_batch, combine, batch_size=10, flush_timeout=0
            )
        )

        # Just one batch at the end (heartbeat at end flushes it)
        assert len(batches_seen) == 1
        assert result == ["a"]

    def test_timeout_flush_respects_item_order(self):
        """Flushed results maintain correct order in output."""
        items: list[tuple[str, list[int]] | None] = [
            ("first", [1]),
            ("second", [2]),
            None,  # flush
            ("third", [3]),
        ]
        order_seen: list[str] = []

        def expand(inp: tuple[str, list[int]]) -> tuple[str, list[int]]:
            return inp

        def process_batch(elements: list[int]) -> list[int]:
            return elements

        def combine(key: str, results: list[int]) -> str:
            order_seen.append(key)
            return key

        result = list(
            batched_expand_map(
                items, expand, process_batch, combine, batch_size=10, flush_timeout=0
            )
        )

        # Order preserved despite partial flush
        assert result == ["first", "second", "third"]
        # combine called in order
        assert order_seen == ["first", "second", "third"]


# =============================================================================
# _StreamingBridge tests
# =============================================================================


class TestStreamingBridge:
    """Tests for _StreamingBridge - the async-to-sync queue bridge."""

    @pytest.fixture
    def bridge(self):
        """Create a bridge with fast heartbeat for testing."""
        from place_layers.server.adapter import _StreamingBridge

        return _StreamingBridge[str, int](heartbeat_interval=0.01)

    @pytest.mark.asyncio
    async def test_submit_and_iterate_basic(self, bridge):
        """Basic flow: submit from async, iterate from sync."""
        loop = asyncio.get_running_loop()
        bridge.set_loop(loop)

        # Submit a request
        future = loop.create_future()
        bridge.submit("request1", future)

        # Iterate should yield the request
        requests = []
        for req in bridge:
            if req is not None:
                requests.append(req)
                break  # Just get one

        assert requests == ["request1"]

    @pytest.mark.asyncio
    async def test_deliver_result_completes_future(self, bridge):
        """deliver_result sets the future's result."""
        loop = asyncio.get_running_loop()
        bridge.set_loop(loop)

        future = loop.create_future()
        bridge.submit("req", future)

        # Consume the request
        for _ in bridge:
            break

        # Deliver result
        bridge.deliver_result(42)

        # Future should complete
        result = await asyncio.wait_for(future, timeout=0.1)
        assert result == 42

    @pytest.mark.asyncio
    async def test_heartbeat_on_timeout(self, bridge):
        """Iterator yields None when queue times out."""
        loop = asyncio.get_running_loop()
        bridge.set_loop(loop)

        # Don't submit anything - iterator should yield None
        heartbeats = []

        def iterate_briefly():
            for i, req in enumerate(bridge):
                if req is None:
                    heartbeats.append(True)
                if i >= 2:  # Get a few iterations
                    break

        t = threading.Thread(target=iterate_briefly)
        t.start()
        t.join(timeout=0.1)

        # Should have received heartbeats
        assert len(heartbeats) > 0

        bridge.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_stops_iteration(self, bridge):
        """shutdown() causes iterator to return."""
        loop = asyncio.get_running_loop()
        bridge.set_loop(loop)

        iteration_stopped = threading.Event()

        def iterate():
            for _ in bridge:
                pass
            iteration_stopped.set()

        t = threading.Thread(target=iterate)
        t.start()

        # Give iterator time to start
        await asyncio.sleep(0.02)

        # Shutdown
        bridge.shutdown()

        # Wait for iteration to stop
        t.join(timeout=0.5)
        assert iteration_stopped.is_set()

    @pytest.mark.asyncio
    async def test_deliver_exception_fails_all_futures(self, bridge):
        """deliver_exception sets exception on all pending futures."""
        loop = asyncio.get_running_loop()
        bridge.set_loop(loop)

        # Submit multiple requests
        futures = [loop.create_future() for _ in range(3)]
        for i, f in enumerate(futures):
            bridge.submit(f"req{i}", f)

        # Consume requests (but don't deliver results)
        count = 0
        for _ in bridge:
            count += 1
            if count >= 3:
                break

        # Deliver exception
        bridge.deliver_exception(ValueError("backend failed"))

        # All futures should fail
        for f in futures:
            with pytest.raises(ValueError, match="backend failed"):
                await asyncio.wait_for(f, timeout=0.1)

    @pytest.mark.asyncio
    async def test_fifo_order_preserved(self, bridge):
        """Results are delivered to futures in FIFO order."""
        loop = asyncio.get_running_loop()
        bridge.set_loop(loop)

        # Submit requests in order
        futures = [loop.create_future() for _ in range(3)]
        for i, f in enumerate(futures):
            bridge.submit(f"req{i}", f)

        # Consume all requests
        requests = []
        for req in bridge:
            if req is not None:
                requests.append(req)
            if len(requests) >= 3:
                break

        assert requests == ["req0", "req1", "req2"]

        # Deliver results in same order
        bridge.deliver_result(100)
        bridge.deliver_result(200)
        bridge.deliver_result(300)

        # Each future gets its corresponding result
        results = [await asyncio.wait_for(f, timeout=0.1) for f in futures]
        assert results == [100, 200, 300]

    @pytest.mark.asyncio
    async def test_concurrent_submits(self, bridge):
        """Multiple async tasks can submit concurrently."""
        loop = asyncio.get_running_loop()
        bridge.set_loop(loop)

        received = []
        iteration_done = threading.Event()

        def iterate():
            for req in bridge:
                if req is not None:
                    received.append(req)
                if len(received) >= 5:
                    break
            iteration_done.set()

        t = threading.Thread(target=iterate)
        t.start()

        # Submit from multiple concurrent tasks
        async def submit_task(name: str):
            await asyncio.sleep(0.001)  # Slight stagger
            future = loop.create_future()
            bridge.submit(name, future)
            return future

        futures = await asyncio.gather(
            submit_task("a"),
            submit_task("b"),
            submit_task("c"),
            submit_task("d"),
            submit_task("e"),
        )

        # Wait for all to be received
        iteration_done.wait(timeout=0.5)

        # All requests received (order may vary due to concurrency)
        assert sorted(received) == ["a", "b", "c", "d", "e"]

        bridge.shutdown()
        t.join(timeout=0.5)


# =============================================================================
# Integration: SyncBackendAdapter with mock backend
# =============================================================================


class TestSyncBackendAdapterIntegration:
    """Integration tests for SyncBackendAdapter with mock backend."""

    @pytest.mark.asyncio
    async def test_generate_one_basic(self):
        """Basic generate_one call through adapter."""
        from unittest.mock import MagicMock

        from place_layers.backend.types import GenerateRequest
        from place_layers.clients.types import GeneratedSample, GenerationResult
        from place_layers.server.adapter import SyncBackendAdapter

        # Create mock backend
        mock_backend = MagicMock()
        mock_backend.__enter__ = MagicMock(return_value=mock_backend)
        mock_backend.__exit__ = MagicMock(return_value=None)

        # Mock generate to yield results for each request
        def mock_generate(requests):
            for req in requests:
                if req is None:
                    continue  # Skip heartbeats
                yield GenerationResult(
                    request_id=req.key,
                    samples=[
                        GeneratedSample(
                            text=f"response for {req.key}",
                            token_ids=[1, 2, 3],
                            token_logprobs=[-0.1, -0.2, -0.3],
                            truncated=False,
                        )
                    ],
                    prompt_token_ids=[100, 101],
                )

        mock_backend.generate = mock_generate
        mock_backend.score_tokens = MagicMock(return_value=iter([]))
        mock_backend.score_text = MagicMock(return_value=iter([]))
        mock_backend.score_chat = MagicMock(return_value=iter([]))

        # Use adapter
        adapter = SyncBackendAdapter(mock_backend, heartbeat_interval=0.01)

        async with adapter:
            request = GenerateRequest(
                key="test-1",
                messages=[],
                n=1,
                max_tokens=100,
                temperature=0.0,
                top_p=1.0,
            )
            result = await adapter.generate_one(request)

            assert result.request_id == "test-1"
            assert len(result.samples) == 1
            assert result.samples[0].text == "response for test-1"

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self):
        """Multiple concurrent generate_one calls."""
        from unittest.mock import MagicMock

        from place_layers.backend.types import GenerateRequest
        from place_layers.clients.types import GeneratedSample, GenerationResult
        from place_layers.server.adapter import SyncBackendAdapter

        mock_backend = MagicMock()
        mock_backend.__enter__ = MagicMock(return_value=mock_backend)
        mock_backend.__exit__ = MagicMock(return_value=None)

        def mock_generate(requests):
            for req in requests:
                if req is None:
                    continue
                yield GenerationResult(
                    request_id=req.key,
                    samples=[
                        GeneratedSample(
                            text=f"result-{req.key}",
                            token_ids=[1],
                            token_logprobs=[-0.1],
                            truncated=False,
                        )
                    ],
                    prompt_token_ids=[100],
                )

        mock_backend.generate = mock_generate
        mock_backend.score_tokens = MagicMock(return_value=iter([]))
        mock_backend.score_text = MagicMock(return_value=iter([]))
        mock_backend.score_chat = MagicMock(return_value=iter([]))

        adapter = SyncBackendAdapter(mock_backend, heartbeat_interval=0.01)

        async with adapter:
            # Submit 5 concurrent requests
            requests = [
                GenerateRequest(
                    key=f"req-{i}",
                    messages=[],
                    n=1,
                    max_tokens=100,
                    temperature=0.0,
                    top_p=1.0,
                )
                for i in range(5)
            ]

            tasks = [adapter.generate_one(req) for req in requests]
            results = await asyncio.gather(*tasks)

            # All should complete with correct IDs
            result_ids = sorted(r.request_id for r in results)
            assert result_ids == ["req-0", "req-1", "req-2", "req-3", "req-4"]

    @pytest.mark.asyncio
    async def test_worker_exception_propagates(self):
        """Exception in worker thread propagates to caller."""
        from unittest.mock import MagicMock

        from place_layers.backend.types import GenerateRequest
        from place_layers.server.adapter import SyncBackendAdapter

        mock_backend = MagicMock()
        mock_backend.__enter__ = MagicMock(return_value=mock_backend)
        mock_backend.__exit__ = MagicMock(return_value=None)

        def mock_generate(requests):
            for req in requests:
                if req is None:
                    continue
                raise RuntimeError("GPU exploded")
            yield from ()  # Make it a generator (never reached)

        mock_backend.generate = mock_generate
        mock_backend.score_tokens = MagicMock(return_value=iter([]))
        mock_backend.score_text = MagicMock(return_value=iter([]))
        mock_backend.score_chat = MagicMock(return_value=iter([]))

        adapter = SyncBackendAdapter(mock_backend, heartbeat_interval=0.01)

        async with adapter:
            request = GenerateRequest(
                key="test",
                messages=[],
                n=1,
                max_tokens=100,
                temperature=0.0,
                top_p=1.0,
            )

            with pytest.raises(RuntimeError, match="GPU exploded"):
                await adapter.generate_one(request)


# =============================================================================
# concurrent_expand_map tests
# =============================================================================


@pytest.mark.parametrize("stream", [sync_stream, async_stream])
class TestConcurrentExpandMap:
    """Tests for concurrent_expand_map - async fan-out/fan-in with concurrency control.

    All tests are parametrized to run with both sync and async iterables,
    ensuring both code paths produce identical results.
    """

    @pytest.mark.asyncio
    async def test_basic_expand_and_combine(self, stream):
        """Basic fan-out, concurrent processing, and fan-in."""
        # Input: requests with multiple choices each
        requests = [
            ("req1", ["a", "b", "c"]),
            ("req2", ["x", "y"]),
        ]

        def expand(req: tuple[str, list[str]]) -> tuple[str, list[str]]:
            key, choices = req
            return (key, choices)

        async def process_one(choice: str) -> str:
            await asyncio.sleep(0.01)
            return choice.upper()

        def combine(key: str, results: list[str]) -> tuple[str, list[str]]:
            return (key, results)

        results: list[tuple[str, list[str]]] = []
        async for result in concurrent_expand_map(
            stream(requests), expand, process_one, combine, max_concurrent=10
        ):
            results.append(result)

        # Results should be in input order
        assert results == [
            ("req1", ["A", "B", "C"]),
            ("req2", ["X", "Y"]),
        ]

    @pytest.mark.asyncio
    async def test_respects_max_concurrent(self, stream):
        """Never exceeds max_concurrent active process_one calls."""
        concurrent_count = 0
        max_observed = 0

        requests = [
            ("req1", [1, 2, 3, 4]),
            ("req2", [5, 6, 7, 8]),
        ]

        def expand(req: tuple[str, list[int]]) -> tuple[str, list[int]]:
            return req

        async def process_one(x: int) -> int:
            nonlocal concurrent_count, max_observed
            concurrent_count += 1
            max_observed = max(max_observed, concurrent_count)
            await asyncio.sleep(0.02)
            concurrent_count -= 1
            return x * 10

        def combine(key: str, results: list[int]) -> tuple[str, list[int]]:
            return (key, results)

        results: list[tuple[str, list[int]]] = []
        async for result in concurrent_expand_map(
            stream(requests), expand, process_one, combine, max_concurrent=3
        ):
            results.append(result)

        assert max_observed <= 3
        assert results == [
            ("req1", [10, 20, 30, 40]),
            ("req2", [50, 60, 70, 80]),
        ]

    @pytest.mark.asyncio
    async def test_empty_expansion_yields_immediately(self, stream):
        """Empty expansion yields combined result immediately."""
        requests = [
            ("req1", [1, 2]),
            ("req2", []),  # Empty
            ("req3", [3]),
        ]

        def expand(req: tuple[str, list[int]]) -> tuple[str, list[int]]:
            return req

        async def process_one(x: int) -> int:
            return x * 10

        def combine(key: str, results: list[int]) -> tuple[str, list[int]]:
            return (key, results)

        results: list[tuple[str, list[int]]] = []
        async for result in concurrent_expand_map(
            stream(requests), expand, process_one, combine, max_concurrent=5
        ):
            results.append(result)

        # All results in order, including empty one
        assert results == [
            ("req1", [10, 20]),
            ("req2", []),
            ("req3", [30]),
        ]

    @pytest.mark.asyncio
    async def test_preserves_input_order(self, stream):
        """Results are yielded in input order even if elements complete out of order."""
        requests = [
            ("slow", [1]),  # Will take longer
            ("fast", [2]),  # Will complete first
        ]

        def expand(req: tuple[str, list[int]]) -> tuple[str, list[int]]:
            return req

        async def process_one(x: int) -> int:
            # Item 1 is slow, item 2 is fast
            await asyncio.sleep(0.05 if x == 1 else 0.01)
            return x

        def combine(key: str, results: list[int]) -> tuple[str, list[int]]:
            return (key, results)

        results: list[tuple[str, list[int]]] = []
        async for result in concurrent_expand_map(
            stream(requests), expand, process_one, combine, max_concurrent=5
        ):
            results.append(result)

        # "slow" comes first even though "fast" element completed first
        assert results == [("slow", [1]), ("fast", [2])]

    @pytest.mark.asyncio
    async def test_skip_none_expansion(self, stream):
        """expand returning None skips that input."""
        requests = [
            ("req1", [1]),
            ("skip", [2]),  # Will be skipped
            ("req3", [3]),
        ]

        def expand(req: tuple[str, list[int]]) -> tuple[str, list[int]] | None:
            if req[0] == "skip":
                return None
            return req

        async def process_one(x: int) -> int:
            return x * 10

        def combine(key: str, results: list[int]) -> tuple[str, list[int]]:
            return (key, results)

        results: list[tuple[str, list[int]]] = []
        async for result in concurrent_expand_map(
            stream(requests), expand, process_one, combine, max_concurrent=5
        ):
            results.append(result)

        assert results == [("req1", [10]), ("req3", [30])]

    @pytest.mark.asyncio
    async def test_exception_propagates(self, stream):
        """Exception in process_one propagates correctly."""
        requests = [
            ("req1", [1, 2, 3]),
        ]

        def expand(req: tuple[str, list[int]]) -> tuple[str, list[int]]:
            return req

        async def process_one(x: int) -> int:
            if x == 2:
                raise ValueError("Bad element")
            return x

        def combine(key: str, results: list[int]) -> tuple[str, list[int]]:
            return (key, results)

        with pytest.raises(ValueError, match="Bad element"):
            async for _ in concurrent_expand_map(
                stream(requests), expand, process_one, combine, max_concurrent=5
            ):
                pass

    @pytest.mark.asyncio
    async def test_empty_input(self, stream):
        """Empty input yields no results."""
        call_count = 0

        def expand(req: str) -> tuple[str, list[int]]:
            return (req, [1])

        async def process_one(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x

        def combine(key: str, results: list[int]) -> str:
            return key

        results: list[str] = []
        async for result in concurrent_expand_map(
            stream([]), expand, process_one, combine, max_concurrent=5
        ):
            results.append(result)

        assert results == []
        assert call_count == 0

    @pytest.mark.asyncio
    async def test_single_element_expansion(self, stream):
        """Single element per request works correctly."""
        requests = ["a", "b", "c"]

        def expand(req: str) -> tuple[str, list[str]]:
            return (req, [req])

        async def process_one(x: str) -> str:
            return x.upper()

        def combine(key: str, results: list[str]) -> str:
            return f"{key}:{results[0]}"

        results: list[str] = []
        async for result in concurrent_expand_map(
            stream(requests), expand, process_one, combine, max_concurrent=5
        ):
            results.append(result)

        assert results == ["a:A", "b:B", "c:C"]
