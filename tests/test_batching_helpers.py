"""Tests for batching helper functions.

Tests cover:
- batched_map: Simple 1:1 batching
- batched_expand_map: Fan-out/batch/fan-in with variable expansion sizes
- concurrent_map: Async concurrent processing with bounded concurrency
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

import pytest

from place_layers.backend.base import (
    batched_expand_map,
    batched_map,
    concurrent_map,
)


# =============================================================================
# batched_map tests
# =============================================================================


class TestBatchedMap:
    """Tests for batched_map - simple 1:1 batching."""

    def test_basic_batching(self):
        """Items are grouped into batches of the specified size."""
        items = list(range(10))
        batches_seen: list[list[int]] = []

        def process(batch: list[int]) -> list[int]:
            batches_seen.append(list(batch))
            return [x * 2 for x in batch]

        result = list(batched_map(items, process, batch_size=3))

        assert batches_seen == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        assert result == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

    def test_exact_batch_size_multiple(self):
        """When input length is exact multiple of batch_size, no partial batch."""
        items = list(range(6))
        batches_seen: list[list[int]] = []

        def process(batch: list[int]) -> list[int]:
            batches_seen.append(list(batch))
            return batch

        result = list(batched_map(items, process, batch_size=2))

        assert batches_seen == [[0, 1], [2, 3], [4, 5]]
        assert result == [0, 1, 2, 3, 4, 5]

    def test_empty_input(self):
        """Empty input yields no results and calls process zero times."""
        call_count = 0

        def process(batch: list[int]) -> list[int]:
            nonlocal call_count
            call_count += 1
            return batch

        result = list(batched_map([], process, batch_size=3))

        assert result == []
        assert call_count == 0

    def test_single_item(self):
        """Single item creates one partial batch."""
        batches_seen: list[list[int]] = []

        def process(batch: list[int]) -> list[int]:
            batches_seen.append(list(batch))
            return [x + 100 for x in batch]

        result = list(batched_map([42], process, batch_size=10))

        assert batches_seen == [[42]]
        assert result == [142]

    def test_batch_size_larger_than_input(self):
        """When batch_size > len(items), all items go in one batch."""
        items = [1, 2, 3]
        batches_seen: list[list[int]] = []

        def process(batch: list[int]) -> list[int]:
            batches_seen.append(list(batch))
            return batch

        result = list(batched_map(items, process, batch_size=100))

        assert batches_seen == [[1, 2, 3]]
        assert result == [1, 2, 3]

    def test_batch_size_one(self):
        """batch_size=1 processes items individually."""
        items = [10, 20, 30]
        batches_seen: list[list[int]] = []

        def process(batch: list[int]) -> list[int]:
            batches_seen.append(list(batch))
            return [x + 1 for x in batch]

        result = list(batched_map(items, process, batch_size=1))

        assert batches_seen == [[10], [20], [30]]
        assert result == [11, 21, 31]

    def test_preserves_order(self):
        """Results maintain input order."""
        items = ["a", "b", "c", "d", "e"]

        def process(batch: list[str]) -> list[str]:
            return [s.upper() for s in batch]

        result = list(batched_map(items, process, batch_size=2))

        assert result == ["A", "B", "C", "D", "E"]

    def test_works_with_generator_input(self):
        """Accepts generator as input (doesn't require list)."""

        def gen():
            for i in range(5):
                yield i

        batches_seen: list[list[int]] = []

        def process(batch: list[int]) -> list[int]:
            batches_seen.append(list(batch))
            return batch

        result = list(batched_map(gen(), process, batch_size=2))

        assert batches_seen == [[0, 1], [2, 3], [4]]
        assert result == [0, 1, 2, 3, 4]

    def test_process_can_return_generator(self):
        """Process function can return any iterable, not just list."""

        def process(batch: list[int]) -> Iterable[int]:
            for x in batch:
                yield x * 2

        result = list(batched_map([1, 2, 3, 4], process, batch_size=2))

        assert result == [2, 4, 6, 8]


# =============================================================================
# batched_expand_map tests
# =============================================================================


class TestBatchedExpandMap:
    """Tests for batched_expand_map - fan-out/batch/fan-in pattern."""

    def test_basic_expansion(self):
        """Basic case: each input expands to multiple elements."""
        inputs = [
            ("req0", ["a", "b"]),
            ("req1", ["c", "d", "e"]),
        ]
        batches_seen: list[list[str]] = []

        def expand(inp: tuple[str, list[str]]) -> tuple[str, list[str]]:
            key, elements = inp
            return (key, elements)

        def process_batch(elements: list[str]) -> list[str]:
            batches_seen.append(list(elements))
            return [e.upper() for e in elements]

        def combine(key: str, results: list[str]) -> tuple[str, tuple[str, ...]]:
            return (key, tuple(results))

        result = list(
            batched_expand_map(inputs, expand, process_batch, combine, batch_size=4)
        )

        assert batches_seen == [["a", "b", "c", "d"], ["e"]]
        assert result == [("req0", ("A", "B")), ("req1", ("C", "D", "E"))]

    def test_empty_expansion_preserves_order(self):
        """Empty expansions are yielded in correct order."""
        inputs = [
            ("req0", ["a", "b"]),
            ("req1", []),  # empty
            ("req2", ["c"]),
        ]

        def expand(inp: tuple[str, list[str]]) -> tuple[str, list[str]]:
            return inp

        def process_batch(elements: list[str]) -> list[str]:
            return [e.upper() for e in elements]

        def combine(key: str, results: list[str]) -> tuple[str, int]:
            return (key, len(results))

        result = list(
            batched_expand_map(inputs, expand, process_batch, combine, batch_size=10)
        )

        # Order must be preserved: req0, req1, req2
        assert result == [("req0", 2), ("req1", 0), ("req2", 1)]

    def test_single_request_spans_multiple_batches(self):
        """One input with many elements spans multiple batches."""
        inputs = [("req0", list(range(10)))]
        batches_seen: list[list[int]] = []

        def expand(inp: tuple[str, list[int]]) -> tuple[str, list[int]]:
            return inp

        def process_batch(elements: list[int]) -> list[int]:
            batches_seen.append(list(elements))
            return [e * 2 for e in elements]

        def combine(key: str, results: list[int]) -> tuple[str, list[int]]:
            return (key, results)

        result = list(
            batched_expand_map(inputs, expand, process_batch, combine, batch_size=4)
        )

        # Elements split across batches
        assert batches_seen == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
        # But combined back into single result
        assert result == [("req0", [0, 2, 4, 6, 8, 10, 12, 14, 16, 18])]

    def test_mixed_sizes_with_empty(self):
        """Mix of empty, small, and large expansions."""
        inputs = [
            ("a", ["x", "y"]),  # 2 elements
            ("b", []),  # empty
            ("c", ["1", "2", "3", "4", "5", "6"]),  # 6 elements (spans 2 batches)
            ("d", ["z"]),  # 1 element
        ]
        batches_seen: list[list[str]] = []

        def expand(inp: tuple[str, list[str]]) -> tuple[str, list[str]]:
            return inp

        def process_batch(elements: list[str]) -> list[str]:
            batches_seen.append(list(elements))
            return elements

        def combine(key: str, results: list[str]) -> tuple[str, int]:
            return (key, len(results))

        result = list(
            batched_expand_map(inputs, expand, process_batch, combine, batch_size=4)
        )

        # Batches: [x,y,1,2], [3,4,5,6], [z]
        assert batches_seen == [["x", "y", "1", "2"], ["3", "4", "5", "6"], ["z"]]
        # Results in order with correct counts
        assert result == [("a", 2), ("b", 0), ("c", 6), ("d", 1)]

    def test_none_return_skips_item(self):
        """When expand returns None, item is skipped entirely."""
        inputs = [("keep1", [1]), ("skip", [2]), ("keep2", [3])]

        def expand(
            inp: tuple[str, list[int]],
        ) -> tuple[str, list[int]] | None:
            key, elements = inp
            if key == "skip":
                return None
            return (key, elements)

        def process_batch(elements: list[int]) -> list[int]:
            return elements

        def combine(key: str, results: list[int]) -> str:
            return key

        result = list(
            batched_expand_map(inputs, expand, process_batch, combine, batch_size=10)
        )

        assert result == ["keep1", "keep2"]

    def test_all_empty_expansions(self):
        """All inputs have empty expansions."""
        inputs = [("a", []), ("b", []), ("c", [])]

        def expand(inp: tuple[str, list[Any]]) -> tuple[str, list[Any]]:
            return inp

        def process_batch(elements: list[Any]) -> list[Any]:
            raise AssertionError("Should not be called for empty expansions")

        def combine(key: str, results: list[Any]) -> str:
            assert results == []
            return key

        result = list(
            batched_expand_map(inputs, expand, process_batch, combine, batch_size=10)
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
            batched_expand_map([], expand, process_batch, combine, batch_size=10)
        )

        assert result == []
        assert call_count == 0

    def test_exact_batch_boundaries(self):
        """Elements align exactly with batch boundaries."""
        inputs = [
            ("a", [1, 2, 3, 4]),  # exactly batch_size
            ("b", [5, 6, 7, 8]),  # exactly batch_size
        ]
        batches_seen: list[list[int]] = []

        def expand(inp: tuple[str, list[int]]) -> tuple[str, list[int]]:
            return inp

        def process_batch(elements: list[int]) -> list[int]:
            batches_seen.append(list(elements))
            return elements

        def combine(key: str, results: list[int]) -> tuple[str, list[int]]:
            return (key, results)

        result = list(
            batched_expand_map(inputs, expand, process_batch, combine, batch_size=4)
        )

        assert batches_seen == [[1, 2, 3, 4], [5, 6, 7, 8]]
        assert result == [("a", [1, 2, 3, 4]), ("b", [5, 6, 7, 8])]

    def test_result_order_within_expansion(self):
        """Results within an expansion maintain element order."""
        inputs = [("req", list(range(7)))]
        batches_seen: list[list[int]] = []

        def expand(inp: tuple[str, list[int]]) -> tuple[str, list[int]]:
            return inp

        def process_batch(elements: list[int]) -> list[int]:
            batches_seen.append(list(elements))
            # Return in same order
            return [e * 10 for e in elements]

        def combine(key: str, results: list[int]) -> list[int]:
            return results

        result = list(
            batched_expand_map(inputs, expand, process_batch, combine, batch_size=3)
        )

        assert batches_seen == [[0, 1, 2], [3, 4, 5], [6]]
        # Results recombined in original order
        assert result == [[0, 10, 20, 30, 40, 50, 60]]

    def test_generator_input(self):
        """Works with generator input (doesn't require list)."""

        def gen():
            yield ("a", [1, 2])
            yield ("b", [3])

        def expand(inp: tuple[str, list[int]]) -> tuple[str, list[int]]:
            return inp

        def process_batch(elements: list[int]) -> list[int]:
            return elements

        def combine(key: str, results: list[int]) -> tuple[str, list[int]]:
            return (key, results)

        result = list(
            batched_expand_map(gen(), expand, process_batch, combine, batch_size=10)
        )

        assert result == [("a", [1, 2]), ("b", [3])]


# =============================================================================
# concurrent_map tests
# =============================================================================


class TestConcurrentMap:
    """Tests for concurrent_map - async concurrent processing."""

    @pytest.mark.asyncio
    async def test_basic_concurrent(self):
        """Basic concurrent processing of items."""
        items = [1, 2, 3, 4, 5]

        async def process(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        results: list[int] = []
        async for result in concurrent_map(items, process, max_concurrent=3):
            results.append(result)

        # Results may be in any order, but all should be present
        assert sorted(results) == [2, 4, 6, 8, 10]

    @pytest.mark.asyncio
    async def test_respects_max_concurrent(self):
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

        items = list(range(10))
        results: list[int] = []
        async for result in concurrent_map(items, process, max_concurrent=3):
            results.append(result)

        assert max_observed <= 3
        assert sorted(results) == list(range(10))

    @pytest.mark.asyncio
    async def test_empty_input(self):
        """Empty input yields no results."""
        call_count = 0

        async def process(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x

        results: list[int] = []
        async for result in concurrent_map([], process, max_concurrent=5):
            results.append(result)

        assert results == []
        assert call_count == 0

    @pytest.mark.asyncio
    async def test_single_item(self):
        """Single item is processed correctly."""

        async def process(x: int) -> int:
            return x * 10

        results: list[int] = []
        async for result in concurrent_map([42], process, max_concurrent=5):
            results.append(result)

        assert results == [420]

    @pytest.mark.asyncio
    async def test_max_concurrent_one(self):
        """max_concurrent=1 processes items sequentially."""
        execution_order: list[int] = []

        async def process(x: int) -> int:
            execution_order.append(x)
            await asyncio.sleep(0.01)
            return x

        items = [1, 2, 3, 4, 5]
        results: list[int] = []
        async for result in concurrent_map(items, process, max_concurrent=1):
            results.append(result)

        # With max_concurrent=1, should process in order
        assert execution_order == [1, 2, 3, 4, 5]
        assert results == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_max_concurrent_exceeds_items(self):
        """max_concurrent > len(items) processes all concurrently."""
        start_times: dict[int, float] = {}
        loop = asyncio.get_event_loop()

        async def process(x: int) -> int:
            start_times[x] = loop.time()
            await asyncio.sleep(0.05)
            return x

        items = [1, 2, 3]
        results: list[int] = []
        async for result in concurrent_map(items, process, max_concurrent=10):
            results.append(result)

        # All items should start at nearly the same time
        times = list(start_times.values())
        assert max(times) - min(times) < 0.02  # All started within 20ms
        assert sorted(results) == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_generator_input(self):
        """Works with async generator input."""

        def gen():
            for i in range(5):
                yield i

        async def process(x: int) -> int:
            return x * 2

        results: list[int] = []
        async for result in concurrent_map(gen(), process, max_concurrent=2):
            results.append(result)

        assert sorted(results) == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_exception_propagates(self):
        """Exceptions from process function propagate correctly."""

        async def process(x: int) -> int:
            if x == 3:
                raise ValueError("Bad value")
            return x

        with pytest.raises(ValueError, match="Bad value"):
            async for _ in concurrent_map([1, 2, 3, 4], process, max_concurrent=2):
                pass

    @pytest.mark.asyncio
    async def test_results_yielded_as_completed(self):
        """Results are yielded as they complete, not in input order."""
        completion_order: list[int] = []

        async def process(x: int) -> int:
            # Items complete in reverse order
            await asyncio.sleep((5 - x) * 0.01)
            completion_order.append(x)
            return x

        items = [1, 2, 3, 4, 5]
        results: list[int] = []
        async for result in concurrent_map(items, process, max_concurrent=5):
            results.append(result)

        # With varying delays, higher numbers complete first
        # (5 has 0 delay, 1 has 0.04 delay)
        assert results[0] == 5  # Fastest completes first
        assert sorted(results) == [1, 2, 3, 4, 5]
