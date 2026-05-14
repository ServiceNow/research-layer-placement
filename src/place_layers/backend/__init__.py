"""Generation backend infrastructure.

Provides polymorphic backends for model inference:
- VLLMBackend: vLLM engine, local in-process or remote HTTP (AsyncBackend)
- TransformersBackend: Local HuggingFace models (SyncBackend)
- PlacementSerialBackend: Local per-request placement execution (AsyncBackend)
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
    NoStructure,
    PlacedGenerateChatRequest,
    PlacedGenerateTextRequest,
    PlacedScoreChatRequest,
    PlacedScoreTextRequest,
    PlacedScoreTokensRequest,
    ScoreChatRequest,
    ScoreChatResult,
    ScoreTextRequest,
    ScoreTextResult,
    ScoreTokensRequest,
    ScoreTokensResult,
    SupernetStructure,
)

# Import backend configs for discriminated union
from place_layers.backend.mock import MockBackendConfig
from place_layers.backend.placement_serial import PlacementSerialBackendConfig
from place_layers.backend.rest import RESTBackendConfig, RESTDNSBackendConfig
from place_layers.backend.transformers import TransformersBackendConfig
from place_layers.backend.vllm import VLLMBackendConfig

# Discriminated union of all backend configs
BackendConfig = Annotated[
    VLLMBackendConfig
    | TransformersBackendConfig
    | PlacementSerialBackendConfig
    | RESTBackendConfig
    | RESTDNSBackendConfig
    | MockBackendConfig,
    Field(discriminator="backend_type"),
]
