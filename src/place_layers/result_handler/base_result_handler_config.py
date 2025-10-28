"""Base configuration for result handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from place_layers.result_handler.base_result_handler import BaseResultHandler


class BaseResultHandlerConfig(BaseModel):
    """Base configuration for all result handler implementations."""

    model_config = {"arbitrary_types_allowed": True}

    handler_type: str = Field(description="Type identifier for result handler")

    # Each subclass must define this
    _impl_class: ClassVar[type[BaseResultHandler]]

    def create_result_handler(self) -> BaseResultHandler:
        """Smart constructor: instantiate result handler from config."""
        return self._impl_class(config=self)
