"""JSON result handler implementation."""

import logging
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from place_layers.result_handler.base_result_handler import BaseResultHandler
from place_layers.result_handler.base_result_handler_config import (
    BaseResultHandlerConfig,
)


logger = logging.getLogger(__name__)


class JSONResultHandler(BaseResultHandler):
    """Result handler that saves results as JSON."""

    def __init__(self, config: "JSONResultHandlerConfig") -> None:
        super().__init__(config)
        self.config: JSONResultHandlerConfig  # Type narrowing

    async def save(self, result: BaseModel, output_path: Path) -> None:
        """
        Save results as JSON file.

        Args:
            result: Pydantic model to save.
            output_path: Path to JSON output file.
        """
        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use Pydantic's idiomatic serialization
        with output_path.open("w") as f:
            f.write(result.model_dump_json(indent=self.config.indent))

        logger.info(f"Results saved to {output_path}")


class JSONResultHandlerConfig(BaseResultHandlerConfig):
    """Configuration for JSON result handler."""

    _impl_class: ClassVar[type[BaseResultHandler]] = JSONResultHandler

    handler_type: Literal["json"] = "json"

    indent: int = Field(description="Indentation level for JSON output", ge=0)
