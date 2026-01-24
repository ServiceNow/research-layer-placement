"""Client implementations for vLLM inference.

Provides low-level clients for vLLM backends:
- BaseVLLMClient: Abstract base class for vLLM clients
- RemoteVLLMClient: HTTP client for remote vLLM servers
- LocalVLLMClient: In-process vLLM engine client

Note: Transformers backend handles model operations directly without a separate client.

Import concrete client classes directly from their modules to avoid circular imports:
    from place_layers.clients.local_vllm import LocalVLLMClient
    from place_layers.clients.remote_vllm import RemoteVLLMClient
"""

from place_layers.clients.types import (
    GeneratedSample,
    GenerationResult,
    ScoringResult,
    SequenceLogProbs,
)

__all__ = [
    "GeneratedSample",
    "GenerationResult",
    "ScoringResult",
    "SequenceLogProbs",
]
