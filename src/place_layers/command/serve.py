"""Serve command - starts a REST server exposing a local backend.

This command starts a FastAPI server that wraps a local backend
(TransformersBackend or VLLMBackend with local client), allowing multiple GPUs
to be used for parallel evaluation via the REST backend.

Usage:
    # Start transformers server on GPU 0
    CUDA_VISIBLE_DEVICES=0 python -m place_layers +command=serve +backend=transformers +model=apriel2_0_5b_dev

    # Start vLLM server on GPU 1
    CUDA_VISIBLE_DEVICES=1 python -m place_layers +command=serve +backend=local_vllm +model=apriel2_0_5b_dev
"""

from __future__ import annotations

import logging
from typing import Annotated, ClassVar, Literal

from pydantic import Field

from place_layers.backend import (
    TransformersBackendConfig,
    VLLMBackendConfig,
)
from place_layers.clients.local_vllm import LocalVLLMClientConfig
from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig

logger = logging.getLogger(__name__)

# Servable backends - those that can be wrapped in a REST API
ServableBackendConfig = Annotated[
    TransformersBackendConfig | VLLMBackendConfig,
    Field(discriminator="backend_type"),
]


class ServeCommand(BaseCommand):
    """Command to start a REST server exposing a local backend."""

    config: "ServeCommandConfig"

    async def run(self) -> None:
        """Start the REST server.

        The server runs indefinitely until interrupted (Ctrl+C).
        The model is loaded on startup and unloaded on shutdown.
        """
        import uvicorn

        logger.info(f"Starting server on {self.config.host}:{self.config.port}")
        logger.info(f"Backend type: {self.config.backend.backend_type}")

        # Create FastAPI app based on backend type
        from place_layers.server.app import create_app

        if isinstance(self.config.backend, TransformersBackendConfig):
            logger.info(f"Model: {self.config.backend.model_path}")
        elif isinstance(self.config.backend, VLLMBackendConfig):
            # VLLMBackend can be local or remote - only local makes sense for serve
            if not isinstance(self.config.backend.client, LocalVLLMClientConfig):
                raise ValueError(
                    "serve command requires local vLLM client, not remote. "
                    "Use +backend=local_vllm instead of +backend=vllm"
                )
            logger.info(f"Model: {self.config.backend.client.model_path}")
        else:
            raise ValueError(
                f"Unknown backend type: {self.config.backend.backend_type}"
            )

        # Both TransformersBackend and VLLMBackend work with create_app
        app = create_app(self.config.backend)

        # Configure uvicorn with access logging enabled
        config = uvicorn.Config(
            app,
            host=self.config.host,
            port=self.config.port,
            log_level="info",
            access_log=True,
        )

        # Run server
        server = uvicorn.Server(config)
        await server.serve()


class ServeCommandConfig(BaseCommandConfig):
    """Configuration for the serve command."""

    _command_class: ClassVar[type[BaseCommand]] = ServeCommand

    command: Literal["serve"] = "serve"

    backend: ServableBackendConfig = Field(
        description="Backend configuration (transformers or vllm with local client)"
    )
    host: str = Field(
        description="Host address to bind to (e.g., '0.0.0.0' for all interfaces)"
    )
    port: int = Field(description="Port number to listen on", ge=1, le=65535)
