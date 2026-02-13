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
from typing import TYPE_CHECKING, Any, Self

from pydantic import Field

if TYPE_CHECKING:
    from place_layers.backend.base import AsyncBackend, SyncBackend

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

    def __init__(
        self,
        config: "BackendEvaluatorConfig",
        *,
        output_dir: Path,
        shared_backend: AsyncBackend | SyncBackend | None = None,
    ) -> None:
        super().__init__(config)

        # Create or use shared backend (but don't enter context yet if creating new)
        if shared_backend is not None:
            self._backend: AsyncBackend | SyncBackend = shared_backend
            self._owns_backend = False  # Don't manage lifecycle
        else:
            self._backend = config.backend.create_backend()
            self._owns_backend = True  # Manage lifecycle in __aenter__/__aexit__

        # Model structure (populated in __aenter__ from backend)
        self._num_layers: int | None = None
        self._mixer_names: frozenset[str] | None = None

        # Track current placement to avoid redundant set_placement calls
        # None means model doesn't support placements (non-supernet)
        self._current_placement: list[str] | None = None

        # Set up output directory for saving results (injected from caller)
        self._base_output_dir = output_dir

        # TraceWriter is created lazily when placement is known
        # (in _init_trace_writer or _set_placement)
        self._trace_writer: TraceWriter | None = None

    async def __aenter__(self) -> Self:
        """Enter context: initialize backend and get model structure."""
        # Only enter backend context if we own it (not shared)
        if self._owns_backend:
            if isinstance(self._backend, AsyncBackend):
                await self._backend.__aenter__()
            else:
                self._backend.__enter__()

        # Try to get model structure from backend (optional - not all backends support it)
        structure = self._backend.model_structure
        if isinstance(structure, SupernetStructure):
            self._num_layers = structure.num_layers
            self._mixer_names = frozenset(structure.mixer_names)
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
        # Only exit backend context if we own it (not shared)
        if self._owns_backend:
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
    def mixer_names(self) -> frozenset[str]:
        """Available mixer types (e.g., {'attention', 'gdn', 'kda'}).

        This is discovered from the model and used only for validation.
        For ordering, use cost_model.phase_order.
        """
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

        placement_dir = self._base_output_dir / placement_code
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

        Args:
            parameters: Layer placement to set (assignments are mixer names).

        Raises:
            ValueError: If placement has wrong number of assignments or invalid mixer names.
        """
        # Validate placement length
        if len(parameters.assignments) != self.num_layers:
            raise ValueError(
                f"Placement has {len(parameters.assignments)} assignments, "
                f"expected {self.num_layers}"
            )

        # Validate mixer names
        valid_names = set(self.mixer_names)
        for name in parameters.assignments:
            if name not in valid_names:
                raise ValueError(
                    f"Invalid mixer name '{name}'. Valid names: {sorted(valid_names)}"
                )

        # Assignments are already mixer names - convert to list for backend
        placement = list(parameters.assignments)

        # Skip if unchanged
        if placement == self._current_placement:
            return

        # Update placement state
        self._current_placement = placement

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

    def create_evaluator(
        self,
        output_dir: Path | None = None,
        shared_backend: AsyncBackend | SyncBackend | None = None,
    ) -> "BackendEvaluator":  # type: ignore[override]
        """Create evaluator with required output directory.

        Args:
            output_dir: Base output directory for results. Required for BackendEvaluator.
            shared_backend: Optional pre-created backend to share across evaluators.
                If provided, this backend is used instead of creating a new one.

        """
        if output_dir is None:
            raise ValueError("output_dir is required for BackendEvaluator")
        return self._impl_class(
            config=self, output_dir=output_dir, shared_backend=shared_backend
        )
