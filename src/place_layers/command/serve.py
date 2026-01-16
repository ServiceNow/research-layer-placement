"""Serve command - starts a REST server exposing TransformersBackend.

This command starts a FastAPI server that wraps a TransformersBackend,
allowing multiple GPUs to be used for parallel evaluation via the REST
backend.

Usage:
    # Start server on GPU 0
    CUDA_VISIBLE_DEVICES=0 python -m place_layers +command=serve +backend=transformers +model=apriel2_0_5b_dev port=8080

    # Start server on GPU 1
    CUDA_VISIBLE_DEVICES=1 python -m place_layers +command=serve +backend=transformers +model=apriel2_0_5b_dev port=8081
"""

from __future__ import annotations

import logging
from typing import ClassVar, Literal

from pydantic import Field

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.backend.base import (
    TransformersBackendConfig,
)

logger = logging.getLogger(__name__)


class ServeCommand(BaseCommand):
    """Command to start a REST server exposing TransformersBackend."""

    config: "ServeCommandConfig"

    async def run(self) -> None:
        """Start the REST server.

        The server runs indefinitely until interrupted (Ctrl+C).
        The model is loaded on startup and unloaded on shutdown.
        """
        import uvicorn

        from place_layers.server.app import create_app

        logger.info(f"Starting server on {self.config.host}:{self.config.port}")
        logger.info(f"Model: {self.config.backend.model_path}")

        # Create FastAPI app with the backend config
        app = create_app(self.config.backend)

        # Configure uvicorn
        config = uvicorn.Config(
            app,
            host=self.config.host,
            port=self.config.port,
            log_level="info",
            # Disable access logs for less noise during evaluation
            access_log=False,
            # Increase keepalive timeout to prevent premature connection closure
            # Default is 5s, which causes issues when client is idle between tasks
            timeout_keep_alive=120,
        )

        # Run server
        server = uvicorn.Server(config)
        await server.serve()


class ServeCommandConfig(BaseCommandConfig):
    """Configuration for the serve command."""

    _command_class: ClassVar[type[BaseCommand]] = ServeCommand

    command: Literal["serve"] = "serve"

    backend: TransformersBackendConfig = Field(
        description="TransformersBackend configuration"
    )
    host: str = Field(
        description="Host address to bind to (e.g., '0.0.0.0' for all interfaces)"
    )
    port: int = Field(description="Port number to listen on", ge=1, le=65535)
