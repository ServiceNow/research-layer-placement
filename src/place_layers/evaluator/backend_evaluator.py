"""Base class for evaluators that use generation backends.

This module provides the common infrastructure for evaluators that:
- Use a generation backend (vLLM, Transformers, REST)
- Support model structure discovery (num_layers, mixer_names)
- Support placement switching for supernet evaluation
- Stream results to validation.jsonl via TraceWriter

Concrete evaluators inherit from BackendEvaluator and implement
their specific evaluation logic in the evaluate() method.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Self

from hydra.core.hydra_config import HydraConfig
from pydantic import Field

from place_layers.backend import (
    AsyncBackend,
    BackendConfig,
    ChatScoringRequest,
    ChatScoringResult,
    SyncBackend,
    TextScoringRequest,
    TextScoringResult,
    TokenScoringRequest,
    TokenScoringResult,
)
from place_layers.core.types import LayerPlacement
from place_layers.evaluator.base_evaluator import BaseEvaluator, BaseEvaluatorConfig
from place_layers.evaluator.traces.writer import TraceWriter

logger = logging.getLogger(__name__)


class BackendEvaluator(BaseEvaluator[LayerPlacement]):
    """Base class for evaluators that use generation backends.

    Provides common infrastructure:
    - Backend lifecycle management (__aenter__, __aexit__)
    - Model structure discovery (num_layers, mixer_names)
    - Placement validation and setting
    - TraceWriter for streaming results to disk
    - Output directory from Hydra config

    Subclasses must implement:
    - evaluate(): The actual evaluation logic

    Must be used as an async context manager:

        async with config.create_evaluator() as evaluator:
            score = await evaluator.evaluate(placement)
    """

    config: "BackendEvaluatorConfig"

    def __init__(self, config: "BackendEvaluatorConfig") -> None:
        super().__init__(config)

        # Create backend (but don't enter context yet)
        self._backend: AsyncBackend | SyncBackend = config.backend.create_backend()

        # Model structure (populated in __aenter__ from backend)
        self._num_layers: int | None = None
        self._mixer_names: tuple[str, ...] | None = None

        # Track current placement to avoid redundant set_placement calls
        self._current_placement: tuple[str, ...] | None = None

        # Set up output directory for saving results
        # Always use Hydra's output directory as base
        try:
            hydra_cfg = HydraConfig.get()
            self._base_output_dir = Path(hydra_cfg.runtime.output_dir)
        except ValueError:
            # HydraConfig not available (e.g., in tests)
            # Use a default temp directory
            import tempfile
            self._base_output_dir = Path(tempfile.mkdtemp(prefix="place_layers_"))
            logger.warning(
                f"HydraConfig not available, using temp output directory: {self._base_output_dir}"
            )

        # Initialize trace writer with base directory (will be updated per-placement)
        self._trace_writer = TraceWriter(self._base_output_dir)

    async def __aenter__(self) -> Self:
        """Enter context: initialize backend and get model structure."""
        if isinstance(self._backend, AsyncBackend):
            await self._backend.__aenter__()
        else:
            self._backend.__enter__()

        # Try to get model structure from backend (optional - not all backends support it)
        try:
            structure = self._backend.model_structure
            self._num_layers = structure.num_layers
            self._mixer_names = structure.mixer_names
            logger.info(
                f"{self.__class__.__name__} initialized: {self._num_layers} layers, "
                f"{len(self._mixer_names)} mixers, backend={self.config.backend.backend_type}"
            )
        except NotImplementedError:
            # Model structure not available (e.g., vLLM backend)
            # This is fine if we're not doing placement switching
            logger.info(
                f"{self.__class__.__name__} initialized: model structure not available "
                f"(backend={self.config.backend.backend_type}). "
                f"Placement switching will not be supported."
            )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context: cleanup backend resources."""
        if isinstance(self._backend, AsyncBackend):
            await self._backend.__aexit__(exc_type, exc_val, exc_tb)
        else:
            self._backend.__exit__(exc_type, exc_val, exc_tb)

    @property
    def num_layers(self) -> int:
        """Number of layers in the model."""
        if self._num_layers is None:
            raise RuntimeError(
                "num_layers not available - evaluator must be entered first"
            )
        return self._num_layers

    @property
    def num_mixers(self) -> int:
        """Number of mixer types per layer."""
        if self._mixer_names is None:
            raise RuntimeError(
                "num_mixers not available - evaluator must be entered first"
            )
        return len(self._mixer_names)

    @property
    def mixer_names(self) -> tuple[str, ...]:
        """Available mixer types (e.g., ('attention', 'gdn', 'kda'))."""
        if self._mixer_names is None:
            raise RuntimeError(
                "mixer_names not available - evaluator must be entered first"
            )
        return self._mixer_names

    @property
    def _output_dir(self) -> Path:
        """Current output directory for traces.

        During optimization, this returns a placement-specific subdirectory.
        For single evaluations, it returns the base output directory.
        """
        return self._trace_writer.output_dir

    async def _set_placement(self, parameters: LayerPlacement) -> None:
        """Validate and set the mixer placement on the backend.

        Args:
            parameters: Layer placement to set.

        Raises:
            ValueError: If placement has wrong number of assignments.
        """
        # Validate placement length
        if len(parameters.assignments) != self.num_layers:
            raise ValueError(
                f"Placement has {len(parameters.assignments)} assignments, "
                f"expected {self.num_layers}"
            )

        # Convert assignment indices to mixer names
        mixer_names = tuple(self.mixer_names[idx] for idx in parameters.assignments)

        # Skip if unchanged
        if mixer_names == self._current_placement:
            return

        # Create placement-specific output directory and TraceWriter
        # This ensures results from different placements don't get mixed up during optimization
        placement_dir = self._get_placement_output_dir(mixer_names)
        self._trace_writer = TraceWriter(placement_dir)
        logger.debug(f"Created TraceWriter for placement subdirectory: {placement_dir.name}")

        # Skip for single-mixer models (no dynamic switching needed)
        if len(set(self.mixer_names)) > 1:
            if isinstance(self._backend, AsyncBackend):
                await self._backend.set_placement(mixer_names)
            else:
                self._backend.set_placement(mixer_names)
            logger.debug(f"Set placement: {mixer_names[:4]}...")

        self._current_placement = mixer_names

    async def score_chat(
        self,
        requests: Iterable[ChatScoringRequest],
    ) -> AsyncIterator[ChatScoringResult]:
        """Score chat completions, yielding results as they complete.

        Handles sync/async dispatch based on backend type.
        Results may arrive in any order - use result.key to join with input.
        """
        if isinstance(self._backend, AsyncBackend):
            async for result in self._backend.score_chat(requests):
                yield result
        else:
            for result in self._backend.score_chat(requests):
                yield result

    async def score_tokens(
        self,
        requests: Iterable[TokenScoringRequest],
    ) -> AsyncIterator[TokenScoringResult]:
        """Score token sequences, yielding results as they complete.

        Handles sync/async dispatch based on backend type.
        Results may arrive in any order - use result.key to join with input.
        """
        if isinstance(self._backend, AsyncBackend):
            async for result in self._backend.score_tokens(requests):
                yield result
        else:
            for result in self._backend.score_tokens(requests):
                yield result

    async def score_text(
        self,
        requests: Iterable[TextScoringRequest],
    ) -> AsyncIterator[TextScoringResult]:
        """Score text continuations, yielding results as they complete.

        Handles sync/async dispatch based on backend type.
        Results may arrive in any order - use result.key to join with input.
        """
        if isinstance(self._backend, AsyncBackend):
            async for result in self._backend.score_text(requests):
                yield result
        else:
            for result in self._backend.score_text(requests):
                yield result


class BackendEvaluatorConfig(BaseEvaluatorConfig):
    """Base config for evaluators using generation backends (vLLM, Transformers, REST)."""

    backend: BackendConfig = Field(
        description="Generation backend configuration (vllm, transformers, or rest)"
    )
