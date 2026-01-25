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
    batched_expand_map,
    batched_map,
    concurrent_expand_map,
    concurrent_map,
    get_model_path,
)
from place_layers.backend.types import (
    BackendError,
    ChatMessage,
    GenerateChatRequest,
    GenerateRequestBase,
    GenerateTextRequest,
    LogProbResult,
    ModelStructure,
    ScoreChatRequest,
    ScoreChatResult,
    ScoreTextRequest,
    ScoreTextResult,
    ScoreTokensRequest,
    ScoreTokensResult,
)

# Import backend configs for discriminated union
from place_layers.backend.rest import RESTBackendConfig, RESTDNSBackendConfig
from place_layers.backend.transformers import TransformersBackendConfig
from place_layers.backend.vllm import VLLMBackendConfig

# Discriminated union of all backend configs
BackendConfig = Annotated[
    VLLMBackendConfig | TransformersBackendConfig | RESTBackendConfig | RESTDNSBackendConfig,
    Field(discriminator="backend_type"),
]

__all__ = [
    "AsyncBackend",
    "SyncBackend",
    "BaseBackendConfig",
    "BackendConfig",
    "BackendError",
    # Helper functions
    "batched_map",
    "batched_expand_map",
    "concurrent_map",
    "concurrent_expand_map",
    "get_model_path",
    # Types
    "ChatMessage",
    "GenerateChatRequest",
    "GenerateRequestBase",
    "GenerateTextRequest",
    "LogProbResult",
    "ModelStructure",
    # Scoring request/result types
    "ScoreTokensRequest",
    "ScoreTokensResult",
    "ScoreTextRequest",
    "ScoreTextResult",
    "ScoreChatRequest",
    "ScoreChatResult",
    # Config classes
    "VLLMBackendConfig",
    "TransformersBackendConfig",
    "RESTBackendConfig",
    "RESTDNSBackendConfig",
]
