"""Generation backend infrastructure.

Provides polymorphic backends for model inference:
- VLLMBackend: vLLM engine, local in-process or remote HTTP (AsyncBackend)
- TransformersBackend: Local HuggingFace models (SyncBackend)
- RESTBackend: Generic REST API backend (AsyncBackend)
"""

from typing import Annotated

from pydantic import Field

from place_layers.backend.base import (
    AsyncBackend,
    BaseBackendConfig,
    SyncBackend,
)
from place_layers.backend.types import (
    ChatMessage,
    LogProbResult,
    ModelStructure,
)

# Import backend configs for discriminated union
from place_layers.backend.rest import RESTBackendConfig
from place_layers.backend.transformers import TransformersBackendConfig
from place_layers.backend.vllm import VLLMBackendConfig

# Discriminated union of all backend configs
BackendConfig = Annotated[
    VLLMBackendConfig | TransformersBackendConfig | RESTBackendConfig,
    Field(discriminator="backend_type"),
]

__all__ = [
    "AsyncBackend",
    "SyncBackend",
    "BaseBackendConfig",
    "BackendConfig",
    "ChatMessage",
    "LogProbResult",
    "ModelStructure",
    # Config classes
    "VLLMBackendConfig",
    "TransformersBackendConfig",
    "RESTBackendConfig",
]
