"""Client implementations for vLLM inference.

Provides low-level clients for vLLM backends:
- RemoteVLLMClient: HTTP client for remote vLLM servers
- LocalVLLMClient: In-process vLLM engine client

Note: Transformers backend handles model operations directly without a separate client.
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
