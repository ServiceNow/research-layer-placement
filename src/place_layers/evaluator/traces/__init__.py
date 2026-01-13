"""Trace types and I/O infrastructure.

Provides:
- Record types: Trace, Generation, LikelihoodRecord, ISProblemRecord
- TraceWriter: Streaming JSONL writer for records
- TraceDataset: Loading and querying trace datasets
- Summary functions: summarize_generations, summarize_likelihood, summarize_is
"""

from place_layers.evaluator.traces.types import (
    Generation,
    ISProblemRecord,
    LikelihoodRecord,
    SkippedProblem,
    Trace,
    TraceDataset,
    TraceRecord,
)
from place_layers.evaluator.traces.writer import (
    TraceWriter,
    get_completed_problems,
    summarize_generations,
    summarize_is,
    summarize_likelihood,
)
