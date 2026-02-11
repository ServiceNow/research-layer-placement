"""YAML result handler implementation."""

import logging
from pathlib import Path
from typing import ClassVar, Literal

import yaml
from pydantic import BaseModel, Field

from place_layers.result_handler.base_result_handler import BaseResultHandler
from place_layers.result_handler.base_result_handler_config import (
    BaseResultHandlerConfig,
)


logger = logging.getLogger(__name__)


class YAMLResultHandler(BaseResultHandler):
    """Result handler that saves results as YAML."""

    def __init__(self, config: "YAMLResultHandlerConfig") -> None:
        super().__init__(config)
        self.config: YAMLResultHandlerConfig  # Type narrowing

    async def save(self, result: BaseModel, output_path: Path) -> None:
        """
        Save results as YAML file.

        Args:
            result: Pydantic model to save.
            output_path: Path to YAML output file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = result.model_dump()
        with output_path.open("w") as f:
            yaml.dump(
                data,
                f,
                default_flow_style=self.config.default_flow_style,
                sort_keys=self.config.sort_keys,
            )

        logger.info(f"Results saved to {output_path}")


class YAMLResultHandlerConfig(BaseResultHandlerConfig):
    """Configuration for YAML result handler."""

    _impl_class: ClassVar[type[BaseResultHandler]] = YAMLResultHandler

    handler_type: Literal["yaml"] = "yaml"

    default_flow_style: bool = Field(
        description="Use flow style for collections",
    )
    sort_keys: bool = Field(
        description="Sort dictionary keys in output",
    )
