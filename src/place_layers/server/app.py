"""FastAPI application factory for TransformersBackend REST server.

This module creates a FastAPI app that exposes the TransformersBackend
interface over HTTP. The backend is initialized on startup and cleaned
up on shutdown.

Usage:
    from place_layers.server.app import create_app
    from place_layers.backend.base import (
        TransformersBackendConfig,
    )

    config = TransformersBackendConfig(model_path="/path/to/model", ...)
    app = create_app(config)

    # Run with uvicorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException

from place_layers.server.schemas import (
    BatchGenerateRequest,
    ChatLogProbResultSchema,
    ComputeChatLogProbsRequest,
    ComputeChatLogProbsResponse,
    ComputeLogProbsRequest,
    ComputeLogProbsResponse,
    ComputeTextLogProbsRequest,
    ComputeTextLogProbsResponse,
    ErrorResponse,
    GenerateRequest,
    HealthResponse,
    InfoResponse,
    SetPlacementRequest,
)

if TYPE_CHECKING:
    from place_layers.backend.base import (
        TransformersBackend,
        TransformersBackendConfig,
    )

logger = logging.getLogger(__name__)


def create_app(backend_config: "TransformersBackendConfig") -> FastAPI:
    """Create a FastAPI app wrapping a TransformersBackend.

    The backend is initialized when the server starts and cleaned up
    when it shuts down. All endpoints are synchronous since the
    TransformersBackend is synchronous.

    Args:
        backend_config: Configuration for the TransformersBackend.

    Returns:
        FastAPI application ready to be served.
    """
    # Backend instance - initialized in lifespan
    backend: "TransformersBackend | None" = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal backend
        logger.info(f"Loading model from {backend_config.model_path}")
        logger.info(f"  torch_dtype: {backend_config.torch_dtype}")
        logger.info(f"  device_map: {backend_config.device_map}")
        logger.info(f"  batch_size: {backend_config.batch_size}")

        # Create and initialize backend
        backend = backend_config.create_backend()
        backend.__enter__()

        logger.info("Model loaded successfully, server ready")
        yield

        # Cleanup
        logger.info("Shutting down, unloading model...")
        if backend is not None:
            backend.__exit__(None, None, None)
        logger.info("Model unloaded")

    app = FastAPI(
        title="Transformers Backend Server",
        description="REST API for TransformersBackend with mixer placement support",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(status="ok")

    @app.get("/info", response_model=InfoResponse)
    def info() -> InfoResponse:
        """Get server and model information."""
        if backend is None:
            raise HTTPException(status_code=503, detail="Backend not initialized")

        current_placement = None
        if backend._current_placement is not None:
            current_placement = list(backend._current_placement)

        # Get model structure if available (Apriel2 supernet)
        num_layers = None
        mixer_names = None
        if backend._apriel_structure is not None:
            num_layers = backend._apriel_structure.num_layers
            mixer_names = list(backend._apriel_structure.mixer_names)

        return InfoResponse(
            model_path=backend_config.model_path,
            batch_size=backend_config.batch_size,
            supports_placement=backend._apriel_structure is not None,
            current_placement=current_placement,
            num_layers=num_layers,
            mixer_names=mixer_names,
        )

    @app.post("/generate")
    def generate(req: GenerateRequest) -> dict:
        """Generate completions for a single prompt.

        If placement is provided, sets the mixer configuration before generation.
        The placement setting is a no-op if already set to the same value.
        """
        if backend is None:
            raise HTTPException(status_code=503, detail="Backend not initialized")

        try:
            # Set placement if provided
            if req.placement is not None:
                backend.set_placement(req.placement)

            # Convert messages to dict format expected by backend
            messages = [m.model_dump() for m in req.messages]

            # Generate
            result = backend.generate(
                messages=messages,
                n=req.n,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                request_id=req.request_id,
            )

            return result.model_dump()

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("Generation failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/batch_generate")
    def batch_generate(req: BatchGenerateRequest) -> list[dict]:
        """Generate completions for multiple prompts.

        If placement is provided, sets the mixer configuration before generation.
        """
        if backend is None:
            raise HTTPException(status_code=503, detail="Backend not initialized")

        try:
            # Set placement if provided
            if req.placement is not None:
                backend.set_placement(req.placement)

            # Convert messages to dict format
            messages_batch = [
                [m.model_dump() for m in messages] for messages in req.messages_batch
            ]

            # Generate
            results = backend.batch_generate(
                messages_batch=messages_batch,
                n=req.n,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            )

            return [r.model_dump() for r in results]

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("Batch generation failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/compute_log_probs", response_model=ComputeLogProbsResponse)
    def compute_log_probs(req: ComputeLogProbsRequest) -> ComputeLogProbsResponse:
        """Compute log probabilities for a batch of traces.

        If placement is provided, sets the mixer configuration before scoring.
        """
        if backend is None:
            raise HTTPException(status_code=503, detail="Backend not initialized")

        try:
            # Set placement if provided
            if req.placement is not None:
                backend.set_placement(req.placement)

            # Convert traces to tuple format expected by backend
            traces = [
                (tuple(t.prompt_tokens), tuple(t.completion_tokens)) for t in req.traces
            ]

            # Compute log probs
            log_probs = backend.compute_batch_log_probs(traces)

            return ComputeLogProbsResponse(log_probs=log_probs)

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("Log prob computation failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/compute_text_log_probs", response_model=ComputeTextLogProbsResponse)
    def compute_text_log_probs(
        req: ComputeTextLogProbsRequest,
    ) -> ComputeTextLogProbsResponse:
        """Compute log probabilities for text continuations."""
        if backend is None:
            raise HTTPException(status_code=503, detail="Backend not initialized")
        try:
            log_probs = backend.compute_text_log_probs(
                contexts=req.contexts,
                continuations=req.continuations,
            )
            return ComputeTextLogProbsResponse(log_probs=log_probs)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("Text log prob computation failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/compute_chat_log_probs", response_model=ComputeChatLogProbsResponse)
    def compute_chat_log_probs(
        req: ComputeChatLogProbsRequest,
    ) -> ComputeChatLogProbsResponse:
        """Compute log probabilities for chat completions."""
        if backend is None:
            raise HTTPException(status_code=503, detail="Backend not initialized")
        try:
            # Convert chat messages to dict format expected by backend
            chats = [
                (
                    [m.model_dump() for m in context_msgs],
                    [m.model_dump() for m in completion_msgs],
                )
                for context_msgs, completion_msgs in req.chats
            ]
            results = backend.compute_chat_log_probs(chats)
            response_results = [
                ChatLogProbResultSchema(
                    token_ids=list(r.token_ids),
                    logprobs=list(r.logprobs),
                    answer_range=r.answer_range,
                    total_log_prob=r.total_log_prob,
                )
                for r in results
            ]
            return ComputeChatLogProbsResponse(results=response_results)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("Chat log prob computation failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/set_placement")
    def set_placement(req: SetPlacementRequest) -> dict:
        """Set mixer placement for subsequent requests.

        This sets the placement state on the server. All subsequent
        requests will use this placement until it is changed.
        """
        if backend is None:
            raise HTTPException(status_code=503, detail="Backend not initialized")

        try:
            backend.set_placement(req.mixer_names)
            return {"status": "ok", "placement": req.mixer_names}

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("Set placement failed")
            raise HTTPException(status_code=500, detail=str(e))

    return app
