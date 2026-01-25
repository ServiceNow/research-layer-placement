"""FastAPI application factory for backend REST server.

This module creates a FastAPI app that exposes any backend (sync or async)
over HTTP. The backend is wrapped in an adapter that provides:
- For AsyncBackend: thin wrapper, forwards directly
- For SyncBackend: queue-based batching for GPU efficiency

Endpoints mirror backend methods with single-request semantics.
Server-side batching is transparent to clients.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from place_layers.backend.base import AsyncBackend, BaseBackendConfig, SyncBackend
from place_layers.backend.types import (
    GenerateRequest as BackendGenerateRequest,
    ScoreChatRequest as BackendScoreChatRequest,
    ScoreTextRequest as BackendScoreTextRequest,
    ScoreTokensRequest as BackendScoreTokensRequest,
)
from place_layers.server.adapter import (
    BackendAdapter,
    BackendRequestError,
    create_adapter,
)
from place_layers.server.schemas import (
    GenerateRequest,
    HealthResponse,
    InfoResponse,
    ScoreChatRequest,
    ScoreChatResponse,
    ScoreTextRequest,
    ScoreTextResponse,
    ScoreTokensRequest,
    ScoreTokensResponse,
)

logger = logging.getLogger(__name__)


def create_app(
    backend_config: BaseBackendConfig,
    heartbeat_interval: float = 0.01,
) -> FastAPI:
    """Create a FastAPI app wrapping any backend.

    The backend is wrapped in an adapter that handles:
    - AsyncBackend: forwards requests directly (backend handles concurrency)
    - SyncBackend: queues requests and batches them for GPU efficiency

    Args:
        backend_config: Configuration for the backend.
        heartbeat_interval: Heartbeat interval for streaming bridge.

    Returns:
        FastAPI application ready to be served.
    """
    # Adapter instance - initialized in lifespan
    adapter: BackendAdapter | None = None
    backend: AsyncBackend | SyncBackend | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal adapter, backend
        logger.info(f"Initializing backend: {backend_config.backend_type}")

        # Create backend and adapter
        backend = backend_config.create_backend()
        adapter = create_adapter(backend, heartbeat_interval)

        # Initialize
        await adapter.__aenter__()
        logger.info("Backend initialized, server ready")

        yield

        # Cleanup
        logger.info("Shutting down...")
        if adapter is not None:
            await adapter.__aexit__(None, None, None)
        logger.info("Backend shutdown complete")

    app = FastAPI(
        title="Backend REST Server",
        description="REST API for generation backends with batching support",
        version="0.1.0",
        lifespan=lifespan,
    )

    # =========================================================================
    # Health & Info
    # =========================================================================

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(status="ok")

    @app.get("/info", response_model=InfoResponse)
    async def info() -> InfoResponse:
        """Get server and model information."""
        if backend is None:
            raise HTTPException(status_code=503, detail="Backend not initialized")

        # Get model structure if available
        num_layers = None
        mixer_names = None
        supports_placement = False

        try:
            structure = backend.model_structure
            num_layers = structure.num_layers
            mixer_names = list(structure.mixer_names)
            supports_placement = True
        except NotImplementedError:
            pass

        return InfoResponse(
            model_path=getattr(backend_config, "model_path", "unknown"),
            supports_placement=supports_placement,
            current_placement=None,  # TODO: track current placement
            num_layers=num_layers,
            mixer_names=mixer_names,
        )

    # =========================================================================
    # Generation
    # =========================================================================

    @app.post("/generate")
    async def generate(req: GenerateRequest) -> dict:
        """Generate completions for a single request.

        Mirrors backend.generate() with single-request semantics.
        """
        if adapter is None or backend is None:
            raise HTTPException(status_code=503, detail="Backend not initialized")

        try:
            # Set placement if provided
            if req.placement is not None:
                if isinstance(backend, SyncBackend):
                    backend.set_placement(req.placement)
                else:
                    await backend.set_placement(req.placement)

            # Convert to backend request (includes key, messages, and generation params)
            gen_request = BackendGenerateRequest(
                key=req.key,
                messages=req.messages,
                n=req.n,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            )

            # Generate via adapter
            result = await adapter.generate_one(gen_request)
            return result.model_dump()

        except (ValueError, BackendRequestError) as e:
            raise HTTPException(status_code=400, detail=f"bad_request: {e}")
        except Exception as e:
            logger.exception("Generation failed")
            raise HTTPException(status_code=500, detail=str(e))

    # =========================================================================
    # Scoring
    # =========================================================================

    @app.post("/score_tokens", response_model=ScoreTokensResponse)
    async def score_tokens(req: ScoreTokensRequest) -> ScoreTokensResponse:
        """Score a token sequence.

        Mirrors backend.score_tokens() with single-request semantics.
        """
        if adapter is None or backend is None:
            raise HTTPException(status_code=503, detail="Backend not initialized")

        try:
            # Set placement if provided
            if req.placement is not None:
                if isinstance(backend, SyncBackend):
                    backend.set_placement(req.placement)
                else:
                    await backend.set_placement(req.placement)

            # Convert to backend request
            score_request = BackendScoreTokensRequest(
                key=req.key,
                prompt_tokens=req.prompt_tokens,
                completion_tokens=req.completion_tokens,
            )

            # Score via adapter
            result = await adapter.score_tokens_one(score_request)
            return ScoreTokensResponse(key=result.key, log_prob=result.log_prob)

        except (ValueError, BackendRequestError) as e:
            raise HTTPException(status_code=400, detail=f"bad_request: {e}")
        except Exception as e:
            logger.exception("Token scoring failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/score_text", response_model=ScoreTextResponse)
    async def score_text(req: ScoreTextRequest) -> ScoreTextResponse:
        """Score text continuations with diff-based scoring.

        Mirrors backend.score_text() with single-request semantics.
        """
        if adapter is None or backend is None:
            raise HTTPException(status_code=503, detail="Backend not initialized")

        try:
            # Set placement if provided
            if req.placement is not None:
                if isinstance(backend, SyncBackend):
                    backend.set_placement(req.placement)
                else:
                    await backend.set_placement(req.placement)

            # Convert to backend request
            score_request = BackendScoreTextRequest(
                key=req.key,
                context=req.context,
                choices=req.choices,
            )

            # Score via adapter
            result = await adapter.score_text_one(score_request)
            return ScoreTextResponse(
                key=result.key,
                results=result.results,
            )

        except (ValueError, BackendRequestError) as e:
            raise HTTPException(status_code=400, detail=f"bad_request: {e}")
        except Exception as e:
            logger.exception("Text scoring failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/score_chat", response_model=ScoreChatResponse)
    async def score_chat(req: ScoreChatRequest) -> ScoreChatResponse:
        """Score chat completions.

        Mirrors backend.score_chat() with single-request semantics.
        """
        if adapter is None or backend is None:
            raise HTTPException(status_code=503, detail="Backend not initialized")

        try:
            # Set placement if provided
            if req.placement is not None:
                if isinstance(backend, SyncBackend):
                    backend.set_placement(req.placement)
                else:
                    await backend.set_placement(req.placement)

            # Convert to backend request
            score_request = BackendScoreChatRequest(
                key=req.key,
                context_messages=req.context_messages,
                completion_messages=req.completion_messages,
            )

            # Score via adapter
            result = await adapter.score_chat_one(score_request)
            return ScoreChatResponse(
                key=result.key,
                results=result.results,
            )

        except (ValueError, BackendRequestError) as e:
            raise HTTPException(status_code=400, detail=f"bad_request: {e}")
        except Exception as e:
            logger.exception("Chat scoring failed")
            raise HTTPException(status_code=500, detail=str(e))

    return app
