"""FastAPI application factory for vLLM REST server.

This module creates a FastAPI app that wraps a LocalVLLMClient instance,
exposing the same REST API as the TransformersBackend server.

Supports Apriel2 models with runtime mixer placement switching.

Usage:
    from place_layers.server.app_vllm import create_vllm_app
    from place_layers.backend import LocalVLLMBackendConfig

    config = LocalVLLMBackendConfig(model_path="/path/to/model", ...)
    app = create_vllm_app(config)

    # Run with uvicorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException

from place_layers.backend import LocalVLLMBackendConfig
from place_layers.clients.local_vllm import LocalVLLMClient, LocalVLLMClientConfig
from place_layers.server.schemas import (
    BatchGenerateRequest,
    ComputeChatLogProbsRequest,
    ComputeChatLogProbsResponse,
    ComputeLogProbsRequest,
    ComputeLogProbsResponse,
    ComputeTextLogProbsRequest,
    ComputeTextLogProbsResponse,
    GenerateRequest,
    HealthResponse,
    InfoResponse,
    LogProbResultSchema,
)

logger = logging.getLogger(__name__)


def create_vllm_app(backend_config: LocalVLLMBackendConfig) -> FastAPI:
    """Create a FastAPI app wrapping a LocalVLLMClient instance.

    The vLLM async engine is initialized when the server starts and cleaned up
    when it shuts down. Uses truly async generation via AsyncLLM.

    Args:
        backend_config: Configuration for the vLLM backend.

    Returns:
        FastAPI application ready to be served.
    """
    # Create client config from backend config
    client_config = LocalVLLMClientConfig(
        model_path=backend_config.model_path,
        tokenizer_path=backend_config.tokenizer_path,
        torch_dtype=backend_config.torch_dtype,
        trust_remote_code=backend_config.trust_remote_code,
        gpu_memory_utilization=backend_config.gpu_memory_utilization,
        max_model_len=backend_config.max_model_len,
        disable_compile=backend_config.disable_compile,
        enable_prefix_caching=backend_config.enable_prefix_caching,
        max_concurrent_requests=backend_config.max_concurrent_requests,
    )

    # Client instance - initialized in lifespan
    client: LocalVLLMClient | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal client

        logger.info(f"Loading vLLM model from {backend_config.model_path}")
        logger.info(f"  torch_dtype: {backend_config.torch_dtype}")
        logger.info(f"  gpu_memory_utilization: {backend_config.gpu_memory_utilization}")

        # Create and enter the async client
        client = LocalVLLMClient(client_config)
        await client.__aenter__()

        if client.supports_placement:
            logger.info(f"Model supports placement switching: {client.num_layers} layers")
            logger.info(f"  Available mixers: {client.mixer_names}")
            logger.info(f"  Current placement: {client.current_placement}")

        logger.info("vLLM async engine loaded successfully, server ready")
        yield

        # Cleanup
        logger.info("Shutting down, unloading model...")
        await client.__aexit__(None, None, None)
        client = None
        torch.cuda.empty_cache()
        logger.info("Model unloaded")

    app = FastAPI(
        title="vLLM Backend Server",
        description="REST API for vLLM with Apriel2 mixer placement support",
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
        if client is None:
            raise HTTPException(status_code=503, detail="Backend not initialized")

        return InfoResponse(
            model_path=backend_config.model_path,
            batch_size=client.max_concurrent_requests,
            supports_placement=client.supports_placement,
            current_placement=client.current_placement,
            num_layers=client.num_layers,
            mixer_names=list(client.mixer_names) if client.mixer_names else None,
        )

    @app.post("/generate")
    async def generate(req: GenerateRequest) -> dict:
        """Generate completions for a single prompt.

        If placement is provided, sets the mixer configuration before generation.
        """
        if client is None:
            raise HTTPException(status_code=503, detail="Backend not initialized")

        try:
            # Set placement if provided
            if req.placement is not None:
                await client.set_placement(req.placement)

            # Apply chat template to get prompt text
            messages = [m.model_dump() for m in req.messages]
            prompt = client.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Generate using async client
            result = await client.generate(
                prompt,
                n=req.n,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                request_id=req.request_id,
            )

            return result

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("Generation failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/batch_generate")
    async def batch_generate(req: BatchGenerateRequest) -> list[dict]:
        """Generate completions for multiple prompts.

        If placement is provided, sets the mixer configuration before generation.
        Runs generations concurrently using async client.
        """
        if client is None:
            raise HTTPException(status_code=503, detail="Backend not initialized")

        try:
            # Set placement if provided
            if req.placement is not None:
                await client.set_placement(req.placement)

            # Apply chat template to all prompts
            prompts = []
            for messages in req.messages_batch:
                msgs = [m.model_dump() for m in messages]
                prompt = client.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt)

            # Generate concurrently
            import asyncio

            tasks = [
                client.generate(
                    prompt,
                    n=req.n,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                )
                for prompt in prompts
            ]
            results = await asyncio.gather(*tasks)
            return list(results)

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("Batch generation failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/compute_log_probs", response_model=ComputeLogProbsResponse)
    async def compute_log_probs(req: ComputeLogProbsRequest) -> ComputeLogProbsResponse:
        """Compute log probabilities for a batch of traces.

        If placement is provided, sets the mixer configuration before scoring.
        """
        if client is None:
            raise HTTPException(status_code=503, detail="Backend not initialized")

        try:
            # Set placement if provided
            if req.placement is not None:
                await client.set_placement(req.placement)

            # Compute log probs concurrently
            import asyncio

            tasks = [
                client.compute_log_probs(
                    list(trace.prompt_tokens),
                    list(trace.completion_tokens),
                )
                for trace in req.traces
            ]
            log_probs = await asyncio.gather(*tasks)

            return ComputeLogProbsResponse(log_probs=list(log_probs))

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("Log prob computation failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/compute_text_log_probs", response_model=ComputeTextLogProbsResponse)
    async def compute_text_log_probs(
        req: ComputeTextLogProbsRequest,
    ) -> ComputeTextLogProbsResponse:
        """Compute log probabilities for text continuations with diff-based scoring.

        Uses grouped API where each group is a (context, choices) tuple.
        """
        if client is None:
            raise HTTPException(status_code=503, detail="Backend not initialized")

        try:
            # Set placement if provided
            if req.placement is not None:
                await client.set_placement(req.placement)

            # Build list of (context, continuation) pairs
            pairs = []
            for context, choices in req.text_groups:
                for choice in choices:
                    pairs.append((context, choice))

            # Compute log probs concurrently
            import asyncio

            tasks = [
                client.compute_text_log_probs(context, continuation)
                for context, continuation in pairs
            ]
            results_raw = await asyncio.gather(*tasks)

            # Convert to response format
            results = []
            for token_ids, logprobs, answer_range, total in results_raw:
                results.append(
                    LogProbResultSchema(
                        token_ids=list(token_ids),
                        logprobs=list(logprobs),
                        answer_range=answer_range,
                        total_log_prob=total,
                    )
                )

            return ComputeTextLogProbsResponse(results=results)

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("Text log prob computation failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/compute_chat_log_probs", response_model=ComputeChatLogProbsResponse)
    async def compute_chat_log_probs(
        req: ComputeChatLogProbsRequest,
    ) -> ComputeChatLogProbsResponse:
        """Compute log probabilities for chat completions.

        If placement is provided, sets the mixer configuration before scoring.
        """
        if client is None:
            raise HTTPException(status_code=503, detail="Backend not initialized")

        try:
            # Set placement if provided
            if req.placement is not None:
                await client.set_placement(req.placement)

            # Process each chat
            import asyncio

            async def process_chat(context_msgs, completion_msgs):
                # Apply chat template to context
                context = [m.model_dump() for m in context_msgs]
                context_text = client.tokenizer.apply_chat_template(
                    context, tokenize=False, add_generation_prompt=True
                )

                # Apply chat template to full conversation (context + completion)
                full_msgs = context + [m.model_dump() for m in completion_msgs]
                full_text = client.tokenizer.apply_chat_template(
                    full_msgs, tokenize=False, add_generation_prompt=False
                )

                # Compute continuation (full_text after context_text)
                # Note: we use compute_text_log_probs which handles tokenization
                if full_text.startswith(context_text):
                    continuation = full_text[len(context_text) :]
                else:
                    # Fallback: treat completion as raw text after context
                    continuation = "".join(
                        m.model_dump().get("content", "") for m in completion_msgs
                    )

                return await client.compute_text_log_probs(context_text, continuation)

            tasks = [
                process_chat(context_msgs, completion_msgs)
                for context_msgs, completion_msgs in req.chats
            ]
            results_raw = await asyncio.gather(*tasks)

            # Convert to response format
            results = []
            for token_ids, logprobs, answer_range, total in results_raw:
                results.append(
                    LogProbResultSchema(
                        token_ids=list(token_ids),
                        logprobs=list(logprobs),
                        answer_range=answer_range,
                        total_log_prob=total,
                    )
                )

            return ComputeChatLogProbsResponse(results=results)

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("Chat log prob computation failed")
            raise HTTPException(status_code=500, detail=str(e))

    return app
