#!/usr/bin/env python3
"""Helper script to start a Transformers backend REST server for testing.

Usage:
    python tests/start_test_server.py --model-path MODEL --port PORT [--device-map DEVICE_MAP]
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from place_layers.backend.base import TransformersBackendConfig
from place_layers.server.app import create_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Start Transformers backend REST server")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to HuggingFace model (e.g., 'Qwen/Qwen2-0.5B-Instruct')",
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port to run server on",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind to (default: localhost)",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="cpu",
        help="Device map (default: cpu). Use 'auto' for GPU.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)",
    )

    args = parser.parse_args()

    logger.info(f"Starting server with model: {args.model_path}")
    logger.info(f"  Port: {args.port}")
    logger.info(f"  Host: {args.host}")
    logger.info(f"  Device map: {args.device_map}")
    logger.info(f"  Batch size: {args.batch_size}")

    # Create backend config
    config = TransformersBackendConfig(
        model_path=args.model_path,
        device_map=args.device_map,
        torch_dtype="bfloat16" if args.device_map != "cpu" else "float32",
        batch_size=args.batch_size,
    )

    # Create FastAPI app
    app = create_app(config)

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_keep_alive=120,
    )


if __name__ == "__main__":
    main()
