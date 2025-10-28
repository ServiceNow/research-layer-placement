"""Base configuration for search strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from place_layers.search_strategy.base_search_strategy import BaseSearchStrategy


class BaseSearchStrategyConfig(BaseModel):
    """Base configuration for all search strategy implementations."""

    model_config = {"arbitrary_types_allowed": True}

    strategy_type: str = Field(description="Type identifier for search strategy")

    # Each subclass must define this
    _impl_class: ClassVar[type[BaseSearchStrategy]]

    def create_search_strategy(self) -> BaseSearchStrategy:
        """Smart constructor: instantiate search strategy from config."""
        return self._impl_class(config=self)
