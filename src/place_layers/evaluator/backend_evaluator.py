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
from pathlib import Path
from typing import Self, Sequence

from hydra.core.hydra_config import HydraConfig
from pydantic import Field

from place_layers.backend.base import (
    AsyncBackend,
    BackendConfig,
    ChatLogProbResult,
    ChatMessage,
    SyncBackend,
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
        """Initialize evaluator with backend and trace writer.

        Args:
            config: Evaluator configuration with backend settings.
        """
        super().__init__(config)

        # Create backend (but don't enter context yet)
        self._backend: AsyncBackend | SyncBackend = config.backend.create_backend()

        # Model structure (populated in __aenter__ from backend)
        self._num_layers: int | None = None
        self._mixer_names: tuple[str, ...] | None = None

        # Track current placement to avoid redundant set_placement calls
        self._current_placement: tuple[str, ...] | None = None

        # Set up trace writer for saving results
        hydra_cfg = HydraConfig.get()
        self._output_dir = Path(hydra_cfg.runtime.output_dir)
        self._trace_writer = TraceWriter(self._output_dir)

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

        # Skip for single-mixer models (no dynamic switching needed)
        if len(set(self.mixer_names)) > 1:
            if isinstance(self._backend, AsyncBackend):
                await self._backend.set_placement(mixer_names)
            else:
                self._backend.set_placement(mixer_names)
            logger.debug(f"Set placement: {mixer_names[:4]}...")

        self._current_placement = mixer_names

    # =========================================================================
    # Backend Method Wrappers (sync/async dispatch)
    # =========================================================================

    async def _compute_chat_log_probs(
        self,
        chats: Sequence[tuple[Sequence[ChatMessage], Sequence[ChatMessage]]],
    ) -> list[ChatLogProbResult]:
        """Compute log probabilities for chat completions.

        Handles sync/async dispatch based on backend type.

        Args:
            chats: List of (context_messages, completion_messages) tuples.

        Returns:
            List of ChatLogProbResult with token-level detail.
        """
        if isinstance(self._backend, AsyncBackend):
            return await self._backend.compute_chat_log_probs(chats)
        else:
            return self._backend.compute_chat_log_probs(chats)

    async def _compute_batch_log_probs(
        self,
        sequences: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
    ) -> list[float]:
        """Compute log probabilities for token sequences.

        Handles sync/async dispatch based on backend type.

        Args:
            sequences: List of (prompt_tokens, completion_tokens) tuples.

        Returns:
            List of total log probabilities for each sequence.
        """
        if isinstance(self._backend, AsyncBackend):
            return await self._backend.compute_batch_log_probs(sequences)
        else:
            return self._backend.compute_batch_log_probs(sequences)


# =============================================================================
# Configuration
# =============================================================================


class BackendEvaluatorConfig(BaseEvaluatorConfig):
    """Base configuration for evaluators using generation backends.

    This is an intermediate base class - concrete evaluator configs
    should inherit from this and define evaluator_type and _impl_class.

    Provides:
    - backend: Polymorphic backend configuration (vLLM, Transformers, REST)
    """

    backend: BackendConfig = Field(
        description="Generation backend configuration (vllm, transformers, or rest)"
    )
