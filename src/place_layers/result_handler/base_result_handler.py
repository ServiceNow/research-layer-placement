"""Abstract base class for result handlers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from place_layers.result_handler.base_result_handler_config import (
        BaseResultHandlerConfig,
    )


class BaseResultHandler:
    """Abstract base class for result handlers."""

    def __init__(self, config: "BaseResultHandlerConfig") -> None:
        """
        Initialize the result handler with configuration.

        Args:
            config: Result handler configuration.
        """
        self.config = config

    async def save(self, result: BaseModel, output_path: Path) -> None:
        """
        Save results to specified output path.

        Args:
            result: Pydantic model to save (SearchResult, EvaluationResult, etc.).
            output_path: Path where results should be saved.
        """
        raise NotImplementedError
