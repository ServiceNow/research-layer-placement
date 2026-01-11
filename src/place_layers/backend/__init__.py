"""Generation backend infrastructure.

Provides polymorphic backends for model inference:
- TransformersBackend: Local HuggingFace models
- VLLMBackend: Remote vLLM server via OpenAI-compatible API
- RESTBackend: Generic REST API backend
"""

from place_layers.backend.base import (
    AsyncBackend,
    BackendConfig,
    ChatMessage,
    ModelStructure,
    SyncBackend,
)
