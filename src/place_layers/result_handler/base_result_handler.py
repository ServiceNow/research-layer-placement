"""Abstract base class for result handlers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from place_layers.core.types import SearchResult
    from place_layers.result_handler.base_result_handler_config import (
        BaseResultHandlerConfig,
    )


class BaseResultHandler:
    """Abstract base class for result handlers."""

    def __init__(self, config: BaseResultHandlerConfig) -> None:
        """
        Initialize the result handler with configuration.

        Args:
            config: Result handler configuration.
        """
        self.config = config

    async def save(self, result: SearchResult, output_path: Path) -> None:
        """
        Save search results to specified output path.

        Args:
            result: Search results to save.
            output_path: Path where results should be saved.
        """
        raise NotImplementedError
