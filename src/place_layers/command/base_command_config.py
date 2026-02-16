"""Base configuration for commands."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from place_layers.command.base_command import BaseCommand


class BaseCommandConfig(BaseModel, ABC):
    """Base configuration for all commands.

    Each command subclass should:
    1. Set command: Literal["command_name"] = "command_name"
    2. Add command-specific fields
    3. Set _impl_class: ClassVar[type[BaseCommand]] = CommandClass
    """

    # Discriminator field (filled by subclasses with Literal type)
    command: str = Field(description="Command to execute")

    # Command class binding (each subclass must define this)
    _impl_class: ClassVar[type[BaseCommand]]

    def create_command(self) -> BaseCommand:
        """Create command instance from this config.

        Returns:
            Command instance with full config
        """
        return self._impl_class(config=self)
