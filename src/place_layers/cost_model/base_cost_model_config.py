"""Base configuration for cost models."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from place_layers.cost_model.base_cost_model import BaseCostModel


class BaseCostModelConfig(BaseModel):
    """Base configuration for all cost model implementations."""

    model_config = {"arbitrary_types_allowed": True}

    cost_model_type: str = Field(description="Type identifier for cost model")

    # Each subclass must define this
    _impl_class: ClassVar[type["BaseCostModel"]]

    def create_cost_model(self) -> "BaseCostModel":
        """Smart constructor: instantiate cost model from config."""
        return self._impl_class(config=self)
