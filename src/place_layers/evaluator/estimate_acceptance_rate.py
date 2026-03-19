"""Acceptance rate evaluator for speculative decoding placement exploration.

Estimates speculative decoding acceptance rates between a fixed target
placement and varying draft placements using pre-computed teacher traces.

The target placement is fixed (typically all-attention); the draft placement
is the ``parameters`` argument to ``evaluate()``.

Two caching strategies depending on ``score_metric``:

- **is_step_acceptance_rate** (default, recommended): caches only the target
  probability at each actual completion token — one float per position.
  Memory is negligible (~MB). Only the draft forward pass runs per
  ``evaluate()`` call.

- **token_acceptance_prob** / **step_acceptance_rate**: these need full
  ``(C, V)`` probability vectors from both target and draft.  No caching is
  used; both forward passes run each ``evaluate()`` call.

Three metrics:

1. **Token acceptance probability** (alpha): per-position marginal
   ``alpha_i = sum_v min(p_target(v), p_draft(v))``

2. **Step acceptance rate** (biased): expected fraction of accepted tokens
   per speculative step from marginal alphas.

3. **IS step acceptance rate** (unbiased): Monte Carlo estimate using
   ``P(first k accepted) = E_{x ~ p}[ prod min(q(x_j)/p(x_j), 1) ]``

The ``score_metric`` config field selects which metric ``evaluate()`` returns.

Supports two backends:

- ``TransformersBackend``: All metrics, with optional logits warping
  (temperature, top_p). Requires direct model access for full-vocabulary
  logit tensors.

- ``VLLMBackend``: ``is_step_acceptance_rate`` only, via echo-mode
  per-token logprobs. No logits warping (assumes vLLM handles sampling
  config). Much faster for large models via vLLM's optimized inference.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import torch
import torch.nn.functional as F
from pydantic import Field

from place_layers.evaluator.base_evaluator import BaseEvaluator
from place_layers.evaluator.backend_evaluator import BackendEvaluator, BackendEvaluatorConfig
from place_layers.evaluator.progress import create_progress
from place_layers.evaluator.traces.types import Trace, TraceDataset
from place_layers.core.types import LayerPlacement

if TYPE_CHECKING:
    from place_layers.backend.base import AsyncBackend, SyncBackend
    from place_layers.backend.transformers import TransformersBackend
    from place_layers.backend.vllm import VLLMBackend
    from place_layers.clients.local_vllm import LocalVLLMClient

logger = logging.getLogger(__name__)

ScoreMetric = Literal[
    "token_acceptance_prob",
    "step_acceptance_rate",
    "is_step_acceptance_rate",
]


# ---------------------------------------------------------------------------
# Logits warping (shared with speculative evaluator)
# ---------------------------------------------------------------------------


def _make_warpers(
    temperature: float, top_p: float
) -> "torch.nn.Module | None":
    """Build logits warpers matching the speculative evaluator's sampling."""
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        TemperatureLogitsWarper,
        TopPLogitsWarper,
    )

    warpers = LogitsProcessorList()
    if temperature > 0 and temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(temperature=temperature))
    if 0 < top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p=top_p))
    return warpers if len(warpers) > 0 else None


# ---------------------------------------------------------------------------
# Step acceptance rate estimators
# ---------------------------------------------------------------------------


def _compute_step_acceptance_rates(
    alpha: torch.Tensor,
    gamma: int,
) -> list[float]:
    """Compute step acceptance rate from per-position marginal alphas.

    For each block of gamma positions, the expected accepted fraction is::

        step_rate = (1/gamma) * sum_{k=1}^{gamma} prod_{j=1}^{k} alpha_{j}

    Args:
        alpha: Per-position token acceptance probabilities, shape ``(C,)``.
        gamma: Draft length (block size).

    Returns:
        List of per-block step acceptance rates.
    """
    c_len = alpha.shape[0]
    if c_len == 0:
        return []

    block_rates: list[float] = []
    for s in range(0, c_len, gamma):
        e = min(s + gamma, c_len)
        block_alpha = alpha[s:e]
        cum_prod = torch.cumprod(block_alpha, dim=0)
        block_rates.append(cum_prod.sum().item() / (e - s))

    return block_rates


def _compute_is_step_acceptance_rates_from_probs(
    t_at_token: torch.Tensor,
    d_at_token: torch.Tensor,
    gamma: int,
) -> list[float]:
    """Unbiased IS estimate of step acceptance rate from per-token probs.

    Uses the identity::

        P(A_1, ..., A_k) = E_{x ~ p}[ prod_{j=1}^{k} min(q(x_j)/p(x_j), 1) ]

    where x is sampled from the target p (the teacher trace).

    Args:
        t_at_token: Target probability at actual tokens, shape ``(C,)``.
        d_at_token: Draft probability at actual tokens, shape ``(C,)``.
        gamma: Draft length (block size).

    Returns:
        List of per-block IS step acceptance rates (full blocks only).
    """
    c_len = t_at_token.shape[0]
    if c_len == 0:
        return []

    ratio = torch.clamp(d_at_token / t_at_token.clamp(min=1e-20), max=1.0)

    block_rates: list[float] = []
    for s in range(0, c_len - gamma + 1, gamma):
        block_ratio = ratio[s : s + gamma]
        cum_prod = torch.cumprod(block_ratio, dim=0)
        block_rates.append(cum_prod.sum().item() / gamma)

    return block_rates


# ---------------------------------------------------------------------------
# Forward pass helpers
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _forward_batch(
    traces: list[Trace],
    backend: "TransformersBackend",
    warpers: "torch.nn.Module | None",
) -> tuple[torch.Tensor, list[int], list[int]]:
    """Run a forward pass and return raw logits + per-trace offsets.

    Returns:
        (logits, prompt_lens, completion_lens) where logits is
        ``(B, max_len, V)`` on device.
    """
    device = backend.device
    model = backend.model

    sequences: list[list[int]] = []
    prompt_lens: list[int] = []
    completion_lens: list[int] = []

    for trace in traces:
        seq = list(trace.prompt_tokens) + list(trace.completion_tokens)
        sequences.append(seq)
        prompt_lens.append(len(trace.prompt_tokens))
        completion_lens.append(len(trace.completion_tokens))

    max_len = max(len(s) for s in sequences)
    pad_id = getattr(backend.tokenizer, "pad_token_id", None) or 0

    padded = [s + [pad_id] * (max_len - len(s)) for s in sequences]
    attention_masks = [[1] * len(s) + [0] * (max_len - len(s)) for s in sequences]

    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits

    return logits, prompt_lens, completion_lens


def _extract_completion_probs(
    logits: torch.Tensor,
    prompt_lens: list[int],
    completion_lens: list[int],
    warpers: "torch.nn.Module | None",
    device: torch.device,
) -> list[torch.Tensor]:
    """Extract per-trace completion probability tensors from batch logits.

    Returns list of ``(C_i, V)`` CPU tensors in float32.
    """
    results: list[torch.Tensor] = []
    for i in range(len(prompt_lens)):
        p_len = prompt_lens[i]
        c_len = completion_lens[i]
        start = p_len - 1
        end = p_len + c_len - 1

        trace_logits = logits[i, start:end, :].float()

        if warpers is not None:
            dummy_ids = torch.zeros(c_len, 1, dtype=torch.long, device=device)
            trace_logits = warpers(dummy_ids, trace_logits.clone())

        probs = F.softmax(trace_logits, dim=-1)
        results.append(probs.cpu())

    return results


def _extract_token_probs(
    logits: torch.Tensor,
    traces: list[Trace],
    prompt_lens: list[int],
    completion_lens: list[int],
    warpers: "torch.nn.Module | None",
    device: torch.device,
) -> list[torch.Tensor]:
    """Extract per-trace target probability at actual completion tokens.

    Returns list of ``(C_i,)`` CPU tensors in float32 — compact cache.
    """
    results: list[torch.Tensor] = []
    for i in range(len(prompt_lens)):
        p_len = prompt_lens[i]
        c_len = completion_lens[i]
        start = p_len - 1
        end = p_len + c_len - 1

        trace_logits = logits[i, start:end, :].float()

        if warpers is not None:
            dummy_ids = torch.zeros(c_len, 1, dtype=torch.long, device=device)
            trace_logits = warpers(dummy_ids, trace_logits.clone())

        probs = F.softmax(trace_logits, dim=-1)

        token_ids = torch.tensor(
            list(traces[i].completion_tokens), device=device
        )
        at_token = probs[torch.arange(c_len, device=device), token_ids]
        results.append(at_token.cpu())

    return results


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class AcceptanceRateEvaluator(BackendEvaluator):
    """Evaluator that computes speculative decoding acceptance rates.

    Loads pre-computed teacher traces and computes acceptance rate metrics
    for each draft placement against a fixed target placement.

    When ``score_metric="is_step_acceptance_rate"`` (default), caches only
    the scalar target probability at each actual completion token (~MB).
    Other metrics require full probability vectors and recompute target
    logits each call (2x forward passes per evaluation).
    """

    config: "AcceptanceRateEvaluatorConfig"

    def __init__(
        self,
        config: "AcceptanceRateEvaluatorConfig",
        *,
        output_dir: Path,
        backend: "AsyncBackend | SyncBackend",
    ) -> None:
        super().__init__(config, output_dir=output_dir, backend=backend)
        self._traces: list[Trace] = []
        # Compact cache: t_probs[pos, actual_token] per trace — list of (C_i,) tensors.
        # Used when score_metric == "is_step_acceptance_rate". VLLM path may store None
        # for traces skipped due to missing logprobs (NaN).
        self._target_at_token_cache: list[torch.Tensor | None] | None = None
        self._target_pl: list[str] = []
        self._warpers: torch.nn.Module | None = None
        self._last_metrics: dict[str, float] | None = None

    async def __aenter__(self):
        await super().__aenter__()

        from place_layers.backend.vllm import VLLMBackend

        if isinstance(self._backend, VLLMBackend):
            if self.config.temperature != 1.0:
                raise ValueError(
                    f"temperature={self.config.temperature} is not supported "
                    f"with VLLMBackend (logit warping requires raw logits). "
                    f"Use temperature=1.0 or switch to TransformersBackend."
                )
            if self.config.top_p != 1.0:
                raise ValueError(
                    f"top_p={self.config.top_p} is not supported "
                    f"with VLLMBackend (logit warping requires raw logits). "
                    f"Use top_p=1.0 or switch to TransformersBackend."
                )

        self._traces = self._load_traces()
        self._target_pl = [
            self.config.target_placement[i % len(self.config.target_placement)]
            for i in range(self.num_layers)
        ]
        self._warpers = _make_warpers(self.config.temperature, self.config.top_p)
        logger.info(
            "AcceptanceRateEvaluator: %d traces, gamma=%d, "
            "target=%s, score_metric=%s",
            len(self._traces),
            self.config.gamma,
            self._target_pl[:4],
            self.config.score_metric,
        )
        return self

    def _load_traces(self) -> list[Trace]:
        """Load Trace records from all configured trace source directories."""
        traces: list[Trace] = []
        for source in self.config.trace_sources:
            source_dir = Path(source)
            if not source_dir.exists():
                logger.warning("Trace source not found: %s", source_dir)
                continue
            dataset = TraceDataset.load_from_directory(
                source_dir,
                include_truncated=self.config.include_truncated,
            )
            traces.extend(dataset.traces)
            logger.info(
                "  Loaded %d traces from %s", len(dataset.traces), source_dir
            )
        logger.info("Total: %d traces loaded", len(traces))
        return traces

    @property
    def _needs_full_probs(self) -> bool:
        """Whether the selected score_metric needs full (C, V) prob vectors."""
        return self.config.score_metric in (
            "token_acceptance_prob",
            "step_acceptance_rate",
        )

    # -- Compact cache (IS metric only) -------------------------------------

    def _build_compact_cache(self, backend: "TransformersBackend") -> None:
        """Compute and cache target probability at each actual token.

        Stores one float per completion position per trace — negligible memory.
        """
        logger.info(
            "Caching target token probabilities for %d traces "
            "(compact mode)...",
            len(self._traces),
        )
        backend.set_placement(self._target_pl)
        batch_size = backend.config.batch_size
        self._target_at_token_cache = []

        with create_progress() as progress:
            ptask = progress.add_task(
                "[cyan]caching target probs (compact)",
                total=len(self._traces),
            )
            for start in range(0, len(self._traces), batch_size):
                batch = self._traces[start : start + batch_size]
                logits, p_lens, c_lens = _forward_batch(
                    batch, backend, self._warpers
                )
                batch_at_token = _extract_token_probs(
                    logits, batch, p_lens, c_lens,
                    self._warpers, backend.device,
                )
                self._target_at_token_cache.extend(batch_at_token)
                progress.advance(ptask, len(batch))

        total_floats = sum(t.numel() for t in self._target_at_token_cache)
        mem_mb = total_floats * 4 / 1e6
        logger.info(
            "Target token probs cached: %d traces, %.1f MB",
            len(self._target_at_token_cache), mem_mb,
        )

    # -- IS-only scoring (compact cache) ------------------------------------

    def _score_is_only(
        self,
        backend: "TransformersBackend",
        draft_pl: list[str],
    ) -> float:
        """Score using only IS step acceptance rate with compact cache."""
        backend.set_placement(draft_pl)
        batch_size = backend.config.batch_size
        gamma = self.config.gamma
        all_is_step_rates: list[float] = []

        cache_idx = 0
        with create_progress() as progress:
            ptask = progress.add_task(
                "[cyan]draft acceptance rate (IS)",
                total=len(self._traces),
            )
            for start in range(0, len(self._traces), batch_size):
                batch = self._traces[start : start + batch_size]
                logits, p_lens, c_lens = _forward_batch(
                    batch, backend, self._warpers
                )
                d_at_token_list = _extract_token_probs(
                    logits, batch, p_lens, c_lens,
                    self._warpers, backend.device,
                )

                for i in range(len(batch)):
                    t_at = self._target_at_token_cache[cache_idx].to(  # type: ignore[union-attr]
                        backend.device
                    )
                    d_at = d_at_token_list[i].to(backend.device)
                    cache_idx += 1

                    is_rates = _compute_is_step_acceptance_rates_from_probs(
                        t_at, d_at, gamma
                    )
                    if is_rates:
                        all_is_step_rates.append(
                            sum(is_rates) / len(is_rates)
                        )

                progress.advance(ptask, len(batch))

        score = (
            sum(all_is_step_rates) / len(all_is_step_rates)
            if all_is_step_rates
            else 0.0
        )
        self._last_metrics = {
            "is_step_acceptance_rate": score,
        }
        return score

    # -- Full-probs scoring (no cache, both forward passes) -----------------

    def _score_full_probs(
        self,
        backend: "TransformersBackend",
        draft_pl: list[str],
    ) -> float:
        """Score using full probability vectors (2x forward passes)."""
        batch_size = backend.config.batch_size
        gamma = self.config.gamma

        all_token_probs: list[float] = []
        all_step_rates: list[float] = []
        all_is_step_rates: list[float] = []

        with create_progress() as progress:
            ptask = progress.add_task(
                "[cyan]acceptance rate (full probs)",
                total=len(self._traces),
            )
            for start in range(0, len(self._traces), batch_size):
                batch = self._traces[start : start + batch_size]

                # Target forward
                backend.set_placement(self._target_pl)
                t_logits, p_lens, c_lens = _forward_batch(
                    batch, backend, self._warpers
                )
                t_probs_list = _extract_completion_probs(
                    t_logits, p_lens, c_lens,
                    self._warpers, backend.device,
                )

                # Draft forward
                backend.set_placement(draft_pl)
                d_logits, _, _ = _forward_batch(
                    batch, backend, self._warpers
                )
                d_probs_list = _extract_completion_probs(
                    d_logits, p_lens, c_lens,
                    self._warpers, backend.device,
                )

                for i, trace in enumerate(batch):
                    t_probs = t_probs_list[i].to(backend.device)
                    d_probs = d_probs_list[i].to(backend.device)

                    alpha = torch.min(t_probs, d_probs).sum(dim=-1)
                    all_token_probs.append(float(alpha.mean()))

                    step_rates = _compute_step_acceptance_rates(alpha, gamma)
                    if step_rates:
                        all_step_rates.append(
                            sum(step_rates) / len(step_rates)
                        )

                    c_len = len(trace.completion_tokens)
                    token_idx = torch.tensor(
                        list(trace.completion_tokens), device=backend.device,
                    )
                    t_at = t_probs[
                        torch.arange(c_len, device=backend.device), token_idx
                    ]
                    d_at = d_probs[
                        torch.arange(c_len, device=backend.device), token_idx
                    ]
                    is_rates = _compute_is_step_acceptance_rates_from_probs(
                        t_at, d_at, gamma,
                    )
                    if is_rates:
                        all_is_step_rates.append(
                            sum(is_rates) / len(is_rates)
                        )

                progress.advance(ptask, len(batch))

        self._last_metrics = {
            "token_acceptance_prob": (
                sum(all_token_probs) / len(all_token_probs)
                if all_token_probs
                else 0.0
            ),
            "step_acceptance_rate": (
                sum(all_step_rates) / len(all_step_rates)
                if all_step_rates
                else 0.0
            ),
            "is_step_acceptance_rate": (
                sum(all_is_step_rates) / len(all_is_step_rates)
                if all_is_step_rates
                else 0.0
            ),
        }
        return self._last_metrics[self.config.score_metric]

    # -- vLLM scoring (async, no logits warping) -----------------------------

    async def _get_trace_token_probs_vllm(
        self,
        trace: Trace,
        client: "LocalVLLMClient",
    ) -> torch.Tensor:
        """Get P(actual_token) at each completion position via vLLM echo."""
        logprobs = await client.get_completion_token_logprobs(
            list(trace.prompt_tokens),
            list(trace.completion_tokens),
        )
        return torch.tensor(logprobs, dtype=torch.float32).exp()

    async def _build_compact_cache_vllm(self, backend: "VLLMBackend") -> None:
        """Compute and cache target token probs using vLLM backend."""
        from place_layers.clients.local_vllm import LocalVLLMClient

        client: LocalVLLMClient = backend._client  # type: ignore[assignment]

        logger.info(
            "Caching target token probabilities for %d traces (vLLM)...",
            len(self._traces),
        )
        await backend.set_placement(self._target_pl)
        sem = asyncio.Semaphore(backend.max_concurrent_requests)

        async def _get_one(trace: Trace) -> torch.Tensor:
            async with sem:
                return await self._get_trace_token_probs_vllm(trace, client)

        with create_progress() as progress:
            ptask = progress.add_task(
                "[cyan]caching target probs (vLLM)",
                total=len(self._traces),
            )
            tasks = [asyncio.create_task(_get_one(t)) for t in self._traces]
            self._target_at_token_cache = []
            skipped = 0
            for idx, task in enumerate(tasks):
                result = await task
                if torch.isnan(result).any():
                    logger.warning(
                        "Skipping trace %d: vLLM returned missing logprob(s) (NaN).",
                        idx,
                    )
                    skipped += 1
                    self._target_at_token_cache.append(None)
                else:
                    self._target_at_token_cache.append(result)
                progress.advance(ptask, 1)

        valid_cache = [t for t in self._target_at_token_cache if t is not None]
        if not valid_cache:
            logger.warning(
                "All %d traces skipped due to missing vLLM logprobs (NaN). "
                "No acceptance rate can be computed.",
                len(self._traces),
            )
        total_floats = sum(t.numel() for t in valid_cache)
        mem_mb = total_floats * 4 / 1e6
        logger.info(
            "Target token probs cached: %d traces (%d skipped due to missing logprobs), %.1f MB",
            len(valid_cache),
            skipped,
            mem_mb,
        )

    async def _score_is_only_vllm(
        self,
        backend: "VLLMBackend",
        draft_pl: list[str],
    ) -> float:
        """Score IS step acceptance rate using vLLM backend."""
        from place_layers.clients.local_vllm import LocalVLLMClient

        client: LocalVLLMClient = backend._client  # type: ignore[assignment]

        await backend.set_placement(draft_pl)
        sem = asyncio.Semaphore(backend.max_concurrent_requests)
        gamma = self.config.gamma
        all_is_step_rates: list[float] = []

        async def _get_one(trace: Trace) -> torch.Tensor:
            async with sem:
                return await self._get_trace_token_probs_vllm(trace, client)

        with create_progress() as progress:
            ptask = progress.add_task(
                "[cyan]draft acceptance rate (vLLM IS)",
                total=len(self._traces),
            )
            tasks = [asyncio.create_task(_get_one(t)) for t in self._traces]
            for i, task in enumerate(tasks):
                d_at = await task
                t_at = self._target_at_token_cache[i]  # type: ignore[index]
                if t_at is None:
                    progress.advance(ptask, 1)
                    continue
                if torch.isnan(d_at).any():
                    logger.warning(
                        "Skipping trace %d: vLLM returned missing logprob(s) (NaN) for draft.",
                        i,
                    )
                    progress.advance(ptask, 1)
                    continue

                is_rates = _compute_is_step_acceptance_rates_from_probs(
                    t_at, d_at, gamma
                )
                if is_rates:
                    all_is_step_rates.append(sum(is_rates) / len(is_rates))

                progress.advance(ptask, 1)

        score = (
            sum(all_is_step_rates) / len(all_is_step_rates)
            if all_is_step_rates
            else 0.0
        )
        if not all_is_step_rates:
            logger.warning(
                "No valid sequences for vLLM acceptance rate (all skipped due to missing logprobs). "
                "Returning 0.0."
            )
        self._last_metrics = {"is_step_acceptance_rate": score}
        return score

    # -- Public interface ---------------------------------------------------

    def get_metrics(self) -> dict[str, float] | None:
        """Return acceptance rate metrics from the last evaluation."""
        return self._last_metrics

    async def evaluate(self, parameters: LayerPlacement | None) -> float:
        """Evaluate acceptance rate for a draft placement.

        The *parameters* argument is the **draft** placement. The target
        placement is fixed in the config.

        Supports both TransformersBackend (all metrics, with logits warping)
        and VLLMBackend (is_step_acceptance_rate only, no logits warping).

        Args:
            parameters: Draft layer placement, or None for default.

        Returns:
            The acceptance rate metric selected by ``score_metric``.
        """
        from place_layers.backend.transformers import TransformersBackend
        from place_layers.backend.vllm import VLLMBackend

        if parameters is not None:
            await self._set_placement(parameters)
        draft_pl = (
            list(self._current_placement)
            if self._current_placement
            else list(self._target_pl)
        )

        if isinstance(self._backend, VLLMBackend):
            if self._needs_full_probs:
                raise NotImplementedError(
                    f"score_metric='{self.config.score_metric}' requires full "
                    f"vocabulary probabilities and is only supported with "
                    f"TransformersBackend. Use score_metric='is_step_acceptance_rate' "
                    f"with vLLM."
                )
            if self._target_at_token_cache is None:
                await self._build_compact_cache_vllm(self._backend)
            score = await self._score_is_only_vllm(self._backend, draft_pl)

        elif isinstance(self._backend, TransformersBackend):
            if self._needs_full_probs:
                score = self._score_full_probs(self._backend, draft_pl)
            else:
                if self._target_at_token_cache is None:
                    self._build_compact_cache(self._backend)
                score = self._score_is_only(self._backend, draft_pl)

        else:
            raise TypeError(
                f"AcceptanceRateEvaluator requires TransformersBackend or "
                f"VLLMBackend, got {type(self._backend).__name__}."
            )

        logger.info(
            "Acceptance rate: %s (returning %s=%.4f)",
            ", ".join(
                f"{k}={v:.4f}" for k, v in (self._last_metrics or {}).items()
            ),
            self.config.score_metric,
            score,
        )

        return score


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class AcceptanceRateEvaluatorConfig(BackendEvaluatorConfig):
    """Configuration for acceptance rate evaluator.

    Computes speculative decoding acceptance rates between a fixed target
    placement and the draft placement passed to ``evaluate()``.
    """

    _impl_class: ClassVar[type[BaseEvaluator]] = AcceptanceRateEvaluator

    evaluator_type: Literal["estimate_acceptance_rate"] = "estimate_acceptance_rate"

    trace_sources: list[str] = Field(
        description="Directories containing pre-computed teacher traces "
        "(from precompute_traces command).",
    )

    target_placement: list[str] = Field(
        default=["attention"],
        description="Target placement pattern (cyclically extended to num_layers).",
    )

    gamma: int = Field(
        default=8,
        gt=0,
        description="Draft length (block size) for step acceptance rate estimation.",
    )

    temperature: float = Field(
        default=1.0,
        gt=0,
        description="Sampling temperature applied to logits before computing "
        "acceptance probabilities. Must match the temperature used in "
        "speculative decoding for accurate estimation.",
    )

    top_p: float = Field(
        default=1.0,
        gt=0,
        le=1.0,
        description="Nucleus sampling threshold applied to logits. "
        "Must match the top_p used in speculative decoding.",
    )

    score_metric: ScoreMetric = Field(
        default="is_step_acceptance_rate",
        description="Which metric evaluate() returns as the scalar score: "
        "'token_acceptance_prob', 'step_acceptance_rate', or "
        "'is_step_acceptance_rate' (unbiased, recommended). "
        "is_step_acceptance_rate uses a compact cache (~MB); the others "
        "require 2x forward passes per evaluation (no caching).",
    )

    include_truncated: bool = Field(
        default=False,
        description="If True, include truncated traces (those that hit the "
        "max-token limit during generation). By default these are skipped "
        "because they have incomplete reasoning chains. Useful for "
        "acceptance rate estimation where only token-level probabilities "
        "matter, not answer correctness.",
    )
