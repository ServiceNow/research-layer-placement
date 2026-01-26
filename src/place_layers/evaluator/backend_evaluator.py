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
    BackendError,
    ScoreChatRequest,
    ScoreChatResult,
    ScoreTextRequest,
    ScoreTextResult,
    ScoreTokensRequest,
    ScoreTokensResult,
    SupernetStructure,
    SyncBackend,
    get_model_path,
)
from place_layers.core.types import LayerPlacement
from place_layers.evaluator.base_evaluator import BaseEvaluator, BaseEvaluatorConfig
from place_layers.evaluator.traces.writer import OutputMetadata, TraceWriter

logger = logging.getLogger(__name__)


class BackendEvaluator(BaseEvaluator[LayerPlacement]):
    """Base class for evaluators that use generation backends.

    Provides common infrastructure:
    - Backend lifecycle management (__aenter__, __aexit__)
    - Model structure discovery (num_layers, mixer_names)
    - Placement validation and setting
    - TraceWriter for streaming results to disk (placement-indexed subdirectories)
    - Output directory from Hydra config

    Subclasses must implement:
    - evaluate(): The actual evaluation logic

    Must be used as an async context manager:

        async with config.create_evaluator() as evaluator:
            score = await evaluator.evaluate(placement)

    Output Directory Structure:
        All traces are written to placement-indexed subdirectories:
        - placements/empty/     - For non-supernet or default runs (placement=())
        - placements/a2g3/      - For specific placements (run-length encoded)

        This ensures that different placements never mix results during optimization.
    """

    config: "BackendEvaluatorConfig"

    def __init__(self, config: "BackendEvaluatorConfig") -> None:
        super().__init__(config)

        # Create backend (but don't enter context yet)
        self._backend: AsyncBackend | SyncBackend = config.backend.create_backend()

        # Model structure (populated in __aenter__ from backend)
        self._num_layers: int | None = None
        self._mixer_names: list[str] | None = None

        # Track current placement to avoid redundant set_placement calls
        # None means model doesn't support placements (non-supernet)
        self._current_placement: list[str] | None = None

        # Set up output directory for saving results (from Hydra)
        hydra_cfg = HydraConfig.get()
        self._base_output_dir = Path(hydra_cfg.runtime.output_dir)

        # TraceWriter is created lazily when placement is known
        # (in _init_trace_writer or _set_placement)
        self._trace_writer: TraceWriter | None = None

    async def __aenter__(self) -> Self:
        """Enter context: initialize backend and get model structure."""
        if isinstance(self._backend, AsyncBackend):
            await self._backend.__aenter__()
        else:
            self._backend.__enter__()

        # Try to get model structure from backend (optional - not all backends support it)
        structure = self._backend.model_structure
        if isinstance(structure, SupernetStructure):
            self._num_layers = structure.num_layers
            self._mixer_names = structure.mixer_names
            logger.info(
                f"{self.__class__.__name__} initialized: {self._num_layers} layers, "
                f"{len(self._mixer_names)} mixers, backend={self.config.backend.backend_type}"
            )
        else:
            # Model structure not available (NoStructure)
            # This is fine if we're not doing placement switching
            logger.info(
                f"{self.__class__.__name__} initialized: model structure not available "
                f"(backend={self.config.backend.backend_type}). "
                f"Placement switching will not be supported."
            )

        # Read initial placement from backend (the model's current/default placement)
        self._current_placement = self._backend.placement

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
    def mixer_names(self) -> list[str]:
        """Available mixer types (e.g., ['attention', 'gdn', 'kda'])."""
        if self._mixer_names is None:
            raise RuntimeError(
                "mixer_names not available - evaluator must be entered first"
            )
        return self._mixer_names

    @property
    def _output_dir(self) -> Path:
        """Current output directory for traces.

        Raises:
            RuntimeError: If trace_writer not initialized.
        """
        if self._trace_writer is None:
            raise RuntimeError(
                "_output_dir accessed before trace_writer initialized. "
                "Call _init_trace_writer() first."
            )
        return self._trace_writer.output_dir

    def _build_metadata(self) -> OutputMetadata:
        """Build output metadata from current state.

        Uses _current_placement (None for non-supernet models).
        """
        from place_layers.evaluator.traces.writer import compute_config_hash

        return OutputMetadata(
            model_name=get_model_path(self.config.backend),
            config_hash=compute_config_hash(self.config.model_dump(mode="json")),
            placement=self._current_placement,
        )

    def _get_placement_output_dir(self) -> Path:
        """Get output directory for current placement.

        Creates a subdirectory under base output directory named by a
        run-length encoded representation of the placement.

        For non-supernet models (_current_placement is None), returns
        base output directory (no subdirectory).
        """
        if self._current_placement is None:
            # Non-supernet model: use base output directory
            self._base_output_dir.mkdir(parents=True, exist_ok=True)
            return self._base_output_dir

        placement = self._current_placement

        # Build abbreviation map: first unique letter of each mixer name
        # e.g., {"attention": "a", "gdn": "g", "kda": "k", "mamba": "m"}
        abbrev_map: dict[str, str] = {}
        used_abbrevs: set[str] = set()
        for mixer in sorted(set(placement)):
            # Find first unused character from the mixer name
            for char in mixer.lower():
                if char not in used_abbrevs:
                    abbrev_map[mixer] = char
                    used_abbrevs.add(char)
                    break
            else:
                # Fallback: use first char + number
                base = mixer[0].lower()
                i = 1
                while f"{base}{i}" in used_abbrevs:
                    i += 1
                abbrev_map[mixer] = f"{base}{i}"
                used_abbrevs.add(f"{base}{i}")

        # Run-length encode: [attention, attention, gdn, gdn, gdn] -> "a2g3"
        # Placement is guaranteed non-empty (None case handled above)
        current = placement[0]
        count = 1
        parts: list[str] = []
        for mixer in placement[1:]:
            if mixer == current:
                count += 1
            else:
                parts.append(f"{abbrev_map[current]}{count}")
                current = mixer
                count = 1
        parts.append(f"{abbrev_map[current]}{count}")

        placement_code = "".join(parts)

        # Create subdirectory: placements/<run-length-code>
        placement_dir = self._base_output_dir / "placements" / placement_code
        placement_dir.mkdir(parents=True, exist_ok=True)

        return placement_dir

    def _init_trace_writer(self) -> None:
        """Initialize trace writer for current placement.

        Reads from _current_placement. Must be set before calling.
        """
        output_dir = self._get_placement_output_dir()
        metadata = self._build_metadata()
        self._trace_writer = TraceWriter(output_dir, metadata=metadata)
        logger.debug(f"Initialized TraceWriter: {output_dir.name}")

    async def _set_placement(self, parameters: LayerPlacement) -> None:
        """Validate and set the mixer placement on the backend.

        Also initializes trace writer for the new placement.

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
        placement = [self.mixer_names[idx] for idx in parameters.assignments]

        # Skip if unchanged
        if placement == self._current_placement:
            return

        # Set placement state first (before _init_trace_writer reads it)
        self._current_placement = placement

        # Initialize trace writer for this placement
        self._init_trace_writer()

        # Set placement on backend (skip for single-mixer models)
        if len(set(self.mixer_names)) > 1:
            if isinstance(self._backend, AsyncBackend):
                await self._backend.set_placement(placement)
            else:
                self._backend.set_placement(placement)
            logger.debug(f"Set backend placement: {placement[:4]}...")

    async def score_chat(
        self,
        requests: Iterable[ScoreChatRequest],
    ) -> AsyncIterator[ScoreChatResult | BackendError]:
        """Score chat completions, yielding results as they complete.

        Handles sync/async dispatch based on backend type.
        Results may arrive in any order - use result.key to join with input.

        May yield BackendError for failed requests (e.g., prompt too long).
        Callers must check isinstance(result, BackendError) and write a
        SkippedProblem record to ensure the problem is marked complete.
        """
        if isinstance(self._backend, AsyncBackend):
            async for result in self._backend.score_chat(requests):
                yield result
        else:
            for result in self._backend.score_chat(requests):
                yield result

    async def score_tokens(
        self,
        requests: Iterable[ScoreTokensRequest],
    ) -> AsyncIterator[ScoreTokensResult | BackendError]:
        """Score token sequences, yielding results as they complete.

        Handles sync/async dispatch based on backend type.
        Results may arrive in any order - use result.key to join with input.

        May yield BackendError for failed requests.
        Callers must check isinstance(result, BackendError) and write a
        SkippedProblem record to ensure the problem is marked complete.
        """
        if isinstance(self._backend, AsyncBackend):
            async for result in self._backend.score_tokens(requests):
                yield result
        else:
            for result in self._backend.score_tokens(requests):
                yield result

    async def score_text(
        self,
        requests: Iterable[ScoreTextRequest],
    ) -> AsyncIterator[ScoreTextResult | BackendError]:
        """Score text continuations, yielding results as they complete.

        Handles sync/async dispatch based on backend type.
        Results may arrive in any order - use result.key to join with input.

        May yield BackendError for failed requests.
        Callers must check isinstance(result, BackendError) and write a
        SkippedProblem record to ensure the problem is marked complete.
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
