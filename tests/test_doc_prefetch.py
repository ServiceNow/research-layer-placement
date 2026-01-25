"""Integration tests for TaskBasedEvaluator doc prefetch mechanism.

Tests verify that:
1. _prefetch_docs runs in background and populates _docs_cache
2. _docs_ready events are set after prefetching completes
3. _prepare_task waits for prefetch and returns correct data
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import hydra
import pytest
from hydra.core.global_hydra import GlobalHydra

from place_layers.evaluator.lm_eval_utils import get_docs, load_tasks


@pytest.fixture
def mock_hydra_output(tmp_path: Path) -> Generator[Path, None, None]:
    """Set up mock Hydra runtime environment for testing.

    Patches HydraConfig.get() to return a mock with output directory.
    """
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    mock_hydra_cfg = MagicMock()
    mock_hydra_cfg.runtime.output_dir = str(output_dir)

    with patch(
        "place_layers.evaluator.backend_evaluator.HydraConfig.get",
        return_value=mock_hydra_cfg,
    ):
        yield output_dir


@pytest.fixture
def hydra_init() -> Generator[None, None, None]:
    """Initialize Hydra for config composition."""
    GlobalHydra.instance().clear()
    with hydra.initialize(
        config_path="../src/place_layers/conf", version_base=None
    ):
        yield
    GlobalHydra.instance().clear()


class TestDocPrefetch:
    """Tests for doc prefetching in TaskBasedEvaluator.

    These tests verify the async prefetch mechanism works correctly,
    without needing a real backend (prefetch only uses lm-eval tasks).
    """

    def test_load_tasks_returns_task_dict(self):
        """Test that load_tasks returns a dict of Task objects."""
        tasks = load_tasks(["gsm8k"], default_num_fewshot=0)

        assert "gsm8k" in tasks
        assert tasks["gsm8k"].config.task == "gsm8k"

    def test_get_docs_returns_indexable_docs(self):
        """Test that get_docs returns indexable document collection."""
        tasks = load_tasks(["gsm8k"], default_num_fewshot=0)
        task = tasks["gsm8k"]
        task.download()

        docs = get_docs(task, split="test")

        assert len(docs) > 0
        assert docs[0] is not None

    @pytest.mark.asyncio
    async def test_prefetch_docs_populates_cache(
        self, mock_hydra_output: Path, hydra_init: None
    ):
        """Test that _prefetch_docs populates _docs_cache."""
        from place_layers.backend.mock import MockBackendConfig, ReplayConfig
        from place_layers.evaluator.generative import (
            GenerativeEvaluatorConfig,
        )

        # Create config with mock backend (won't be used for prefetch)
        config = GenerativeEvaluatorConfig(
            backend=MockBackendConfig(
                backend_type="mock",
                num_layers=24,
                mixer_names=("attention", "gdn"),
                max_concurrent_requests=10,
                replay=ReplayConfig(
                    generate_chat=[],
                    generate_text=[],
                    score_tokens=[],
                    score_text=[],
                    score_chat=[],
                ),
            ),
            tasks=["gsm8k"],
            num_fewshot=0,
            max_problems_per_task=5,
            doc_prefetch_count=2,
            samples_per_problem=1,
            max_new_tokens=10,
            temperature=0.0,
            top_p=1.0,
            use_chat_template=True,
            evaluate_truncated_samples=False,
        )

        evaluator = config.create_evaluator()

        # Set up prefetch prerequisites (without entering backend context)
        evaluator._task_cache = load_tasks(["gsm8k"], default_num_fewshot=0)
        for task_name in evaluator._task_cache:
            evaluator._docs_ready[task_name] = threading.Event()

        # Run prefetch
        await evaluator._prefetch_docs(prefetch_ahead=2)

        # Verify docs were cached
        assert "gsm8k" in evaluator._docs_cache
        assert len(evaluator._docs_cache["gsm8k"]) > 0
        assert evaluator._docs_ready["gsm8k"].is_set()

    @pytest.mark.asyncio
    async def test_prefetch_multiple_tasks(
        self, mock_hydra_output: Path, hydra_init: None
    ):
        """Test prefetch with multiple tasks."""
        from place_layers.backend.mock import MockBackendConfig, ReplayConfig
        from place_layers.evaluator.generative import (
            GenerativeEvaluatorConfig,
        )

        config = GenerativeEvaluatorConfig(
            backend=MockBackendConfig(
                backend_type="mock",
                num_layers=24,
                mixer_names=("attention", "gdn"),
                max_concurrent_requests=10,
                replay=ReplayConfig(
                    generate_chat=[],
                    generate_text=[],
                    score_tokens=[],
                    score_text=[],
                    score_chat=[],
                ),
            ),
            tasks=["gsm8k", "arc_easy"],
            num_fewshot=0,
            max_problems_per_task=5,
            doc_prefetch_count=2,
            samples_per_problem=1,
            max_new_tokens=10,
            temperature=0.0,
            top_p=1.0,
            use_chat_template=True,
            evaluate_truncated_samples=False,
        )

        evaluator = config.create_evaluator()
        evaluator._task_cache = load_tasks(
            ["gsm8k", "arc_easy"], default_num_fewshot=0
        )
        for task_name in evaluator._task_cache:
            evaluator._docs_ready[task_name] = threading.Event()

        await evaluator._prefetch_docs(prefetch_ahead=2)

        assert "gsm8k" in evaluator._docs_cache
        assert "arc_easy" in evaluator._docs_cache
        assert evaluator._docs_ready["gsm8k"].is_set()
        assert evaluator._docs_ready["arc_easy"].is_set()

    def test_prepare_task_uses_cached_docs(
        self, mock_hydra_output: Path, hydra_init: None
    ):
        """Test that _prepare_task uses docs from cache."""
        from place_layers.backend.mock import MockBackendConfig, ReplayConfig
        from place_layers.evaluator.generative import (
            GenerativeEvaluatorConfig,
        )
        from place_layers.evaluator.traces.writer import OutputMetadata, TraceWriter

        config = GenerativeEvaluatorConfig(
            backend=MockBackendConfig(
                backend_type="mock",
                num_layers=24,
                mixer_names=("attention", "gdn"),
                max_concurrent_requests=10,
                replay=ReplayConfig(
                    generate_chat=[],
                    generate_text=[],
                    score_tokens=[],
                    score_text=[],
                    score_chat=[],
                ),
            ),
            tasks=["gsm8k"],
            num_fewshot=0,
            max_problems_per_task=5,
            doc_prefetch_count=2,
            samples_per_problem=1,
            max_new_tokens=10,
            temperature=0.0,
            top_p=1.0,
            use_chat_template=True,
            evaluate_truncated_samples=False,
        )

        evaluator = config.create_evaluator()

        # Set up task cache and pre-populate docs cache
        evaluator._task_cache = load_tasks(["gsm8k"], default_num_fewshot=0)
        task = evaluator._task_cache["gsm8k"]
        task.download()
        evaluator._docs_cache["gsm8k"] = get_docs(task, "test")
        evaluator._docs_ready["gsm8k"] = threading.Event()
        evaluator._docs_ready["gsm8k"].set()

        # Create trace writer for _prepare_task
        metadata = OutputMetadata(
            model_name="test",
            placement=None,
            config_hash="test123",
        )
        evaluator._trace_writer = TraceWriter(mock_hydra_output, metadata=metadata)

        # _prepare_task should work with cached docs
        result = evaluator._prepare_task("gsm8k", "generate_until")

        assert result is not None
        returned_task, task_name, docs, remaining = result
        assert task_name == "gsm8k"
        assert len(remaining) <= 5  # max_problems_per_task
