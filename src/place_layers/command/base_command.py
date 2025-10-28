"""Base class for commands."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from place_layers.command.base_command_config import BaseCommandConfig


class CommandError(Exception):
    """Command execution error."""

    pass


class BaseCommand(ABC):
    """Base class for commands."""

    def __init__(self, config: "BaseCommandConfig") -> None:
        """Initialize command.

        Args:
            config: Full configuration (includes command-specific fields)
        """
        self.config = config

    @abstractmethod
    async def run(self) -> None:
        """Execute the command.

        Raises:
            CommandError: If command execution fails
        """
        ...
