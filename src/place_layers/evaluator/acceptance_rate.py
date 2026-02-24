"""Acceptance rate estimator for speculative decoding placement evaluation.

Computes two complementary metrics:

1. **Token acceptance probability** (alpha): the analytical per-position
   probability that a draft token is accepted by rejection sampling::

       alpha_i = sum_v min(p_target(v | x_{<i}), p_draft(v | x_{<i}))
               = 1 - D_TV(p_target, p_draft)

2. **Step acceptance rate**: the expected fraction of draft tokens accepted
   per speculative decoding step of length gamma.  Estimated via importance
   sampling over teacher-generated traces (generated from p_target).

   For a block of gamma tokens x_{1:gamma} sampled from p::

       step_rate = IS_weight * E[#accepted | x] / gamma

   where ``IS_weight = prod q(x_i)/p(x_i)`` corrects for the distribution
   mismatch, and the expected accepted count uses the sequential product of
   per-token acceptance probabilities ``min(1, p(x_i)/q(x_i))``.

Architecture
------------
Inherits from ``TraceBasedEvaluator`` (like ``ImportanceWeightedEvaluator``),
reusing trace loading, task weighting, and resumption infrastructure.

Requires ``TransformersBackend`` for direct model access to obtain
full-vocabulary logit tensors.

Usage
-----
    config = AcceptanceRateEvaluatorConfig(
        trace_sources=["/path/to/target/traces"],
        target_placement=["attention"],
        gamma=8,
    )
    evaluator = config.create_evaluator(output_dir=..., backend=backend)
    async with evaluator:
        score = await evaluator.evaluate(draft_placement)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Sequence

import torch
import torch.nn.functional as F
from pydantic import Field

from place_layers.backend.transformers import TransformersBackend
from place_layers.core.types import LayerPlacement
from place_layers.evaluator.base_evaluator import BaseEvaluator
from place_layers.evaluator.progress import create_progress
from place_layers.evaluator.trace_based_evaluator import (
    ProblemTraces,
    TraceBasedEvaluator,
    TraceBasedEvaluatorConfig,
)
from place_layers.evaluator.traces.types import (
    AcceptanceRateRecord,
    Trace,
)

if TYPE_CHECKING:
    from place_layers.backend.base import AsyncBackend, SyncBackend

logger = logging.getLogger(__name__)


class AcceptanceRateEvaluator(TraceBasedEvaluator):
    """Evaluator that estimates speculative decoding acceptance rates.

    Loads pre-computed teacher traces and scores them with both target
    and draft placements using batched forward passes.  Computes:

    - **token_acceptance_prob**: exact analytical alpha per position from
      full-vocabulary softmax distributions.
    - **step_acceptance_rate**: IS-estimated fraction of accepted tokens
      per speculative step of length gamma.

    Must be used as an async context manager::

        async with config.create_evaluator(output_dir=..., backend=backend) as evaluator:
            score = await evaluator.evaluate(placement)
    """

    config: "AcceptanceRateEvaluatorConfig"

    def __init__(
        self,
        config: "AcceptanceRateEvaluatorConfig",
        *,
        output_dir: Path,
        backend: "AsyncBackend | SyncBackend",
    ) -> None:
        self._target_placement_pattern = config.target_placement
        self._gamma = config.gamma

        super().__init__(config, output_dir=output_dir, backend=backend)

    # ------------------------------------------------------------------
    # Abstract contract: _compute_problem_record (unused in our override
    # of _evaluate_task, but required by TraceBasedEvaluator)
    # ------------------------------------------------------------------

    def _compute_problem_record(
        self,
        problem: ProblemTraces,
        student_log_probs: Sequence[float],
    ) -> Any:
        raise NotImplementedError(
            "AcceptanceRateEvaluator overrides _evaluate_task and does not "
            "use the score_tokens-based _compute_problem_record path."
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _compute_summary(self) -> float:
        """Aggregate acceptance rate records from disk.

        Returns the weighted mean token acceptance probability as the
        primary score.  Also logs step acceptance rate.
        """
        records = self._load_acceptance_records()
        if not records:
            logger.warning("No AcceptanceRateRecords found in output")
            return 0.0

        by_task: dict[str, list[AcceptanceRateRecord]] = {}
        for rec in records:
            by_task.setdefault(rec.task_name, []).append(rec)

        task_token_scores: dict[str, float] = {}
        task_step_scores: dict[str, float] = {}
        for task_name, task_records in sorted(by_task.items()):
            token_mean = sum(
                r.mean_token_acceptance_prob for r in task_records
            ) / len(task_records)
            step_mean = sum(
                r.mean_step_acceptance_rate for r in task_records
            ) / len(task_records)
            task_token_scores[task_name] = token_mean
            task_step_scores[task_name] = step_mean
            logger.info(
                f"  {task_name}: token_acceptance_prob={token_mean:.4f}, "
                f"step_acceptance_rate={step_mean:.4f} "
                f"({len(task_records)} samples)"
            )

        final_token = sum(
            self.task_weights.get(t, 0.0) * s
            for t, s in task_token_scores.items()
        )
        final_step = sum(
            self.task_weights.get(t, 0.0) * s
            for t, s in task_step_scores.items()
        )
        logger.info(
            f"Weighted token_acceptance_prob: {final_token:.4f}, "
            f"step_acceptance_rate: {final_step:.4f}"
        )

        summary_path = self._output_dir / "acceptance_rate_summary.json"
        summary_data = {
            "overall_token_acceptance_prob": round(final_token, 6),
            "overall_step_acceptance_rate": round(final_step, 6),
            "gamma": self._gamma,
            "task_results": {
                t: {
                    "mean_token_acceptance_prob": round(
                        task_token_scores[t], 6
                    ),
                    "mean_step_acceptance_rate": round(
                        task_step_scores[t], 6
                    ),
                    "num_samples": len(by_task[t]),
                    "weight": round(self.task_weights.get(t, 0.0), 6),
                }
                for t in task_token_scores
            },
        }
        summary_path.write_text(json.dumps(summary_data, indent=2))
        logger.info(f"Summary written to {summary_path}")

        return final_token

    def _load_acceptance_records(self) -> list[AcceptanceRateRecord]:
        """Load AcceptanceRateRecord entries from validation.jsonl."""
        from place_layers.evaluator.traces.types import TraceRecordAdapter

        records: list[AcceptanceRateRecord] = []
        jsonl_path = self._output_dir / "validation.jsonl"
        if not jsonl_path.exists():
            return records

        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = TraceRecordAdapter.validate_json(line)
            if isinstance(rec, AcceptanceRateRecord):
                records.append(rec)

        return records

    # ------------------------------------------------------------------
    # Override evaluate to validate backend type
    # ------------------------------------------------------------------

    async def evaluate(self, parameters: LayerPlacement | None) -> float:
        """Evaluate acceptance rates for a draft placement.

        Args:
            parameters: Draft placement.  The target placement is read
                from ``self.config.target_placement``.

        Raises:
            TypeError: If the backend is not ``TransformersBackend``.
        """
        if not isinstance(self._backend, TransformersBackend):
            raise TypeError(
                f"AcceptanceRateEvaluator requires TransformersBackend, "
                f"got {type(self._backend).__name__}. "
                f"Full-vocabulary logit computation needs direct model "
                f"access."
            )
        return await super().evaluate(parameters)

    # ------------------------------------------------------------------
    # Core: batched full-vocab evaluation (overrides TraceBasedEvaluator)
    # ------------------------------------------------------------------

    async def _evaluate_task(
        self,
        task_name: str,
        problems: dict[str, ProblemTraces],
        completed_indices: set[int],
    ) -> None:
        """Score traces with batched forward passes for full-vocab metrics.

        Overrides the ``score_tokens``-based ``_evaluate_task`` from
        ``TraceBasedEvaluator``.
        """
        assert isinstance(self._backend, TransformersBackend)
        backend = self._backend
        batch_size = backend.config.batch_size

        target_pl = [
            self._target_placement_pattern[
                i % len(self._target_placement_pattern)
            ]
            for i in range(self.num_layers)
        ]
        draft_pl = (
            list(self._current_placement)
            if self._current_placement
            else target_pl
        )

        all_traces: list[tuple[ProblemTraces, Trace]] = []
        for pid, problem in problems.items():
            try:
                idx = int(pid.split(":")[-1])
                if idx in completed_indices:
                    continue
            except ValueError:
                pass
            for trace in problem.traces:
                all_traces.append((problem, trace))

        if not all_traces:
            logger.info(
                f"  Task '{task_name}': all problems already completed"
            )
            return

        logger.info(
            f"  Task '{task_name}': scoring {len(all_traces)} traces "
            f"in batches of {batch_size}"
        )

        with create_progress() as progress:
            progress_task = progress.add_task(
                f"[cyan]{task_name}", total=len(all_traces)
            )

            for batch_start in range(0, len(all_traces), batch_size):
                batch = all_traces[batch_start : batch_start + batch_size]
                self._process_batch(
                    batch,
                    target_pl,
                    draft_pl,
                    backend,
                    progress,
                    progress_task,
                )

    @torch.inference_mode()
    def _process_batch(
        self,
        batch: list[tuple[ProblemTraces, Trace]],
        target_pl: list[str],
        draft_pl: list[str],
        backend: TransformersBackend,
        progress: Any,
        progress_task: Any,
    ) -> None:
        """Process one batch: two forward passes, compute both metrics."""
        device = backend.device
        model = backend.model
        gamma = self._gamma

        # -- Build padded input tensors ------------------------------------
        sequences: list[list[int]] = []
        prompt_lens: list[int] = []
        completion_lens: list[int] = []

        for _, trace in batch:
            seq = list(trace.prompt_tokens) + list(trace.completion_tokens)
            sequences.append(seq)
            prompt_lens.append(len(trace.prompt_tokens))
            completion_lens.append(len(trace.completion_tokens))

        max_len = max(len(s) for s in sequences)

        # Left-pad with pad_token_id (fall back to 0)
        pad_id = getattr(backend.tokenizer, "pad_token_id", None) or 0
        padded = []
        attention_masks = []
        pad_lens: list[int] = []
        for seq in sequences:
            pad_len = max_len - len(seq)
            pad_lens.append(pad_len)
            padded.append([pad_id] * pad_len + seq)
            attention_masks.append([0] * pad_len + [1] * len(seq))

        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(
            attention_masks, dtype=torch.long, device=device
        )

        # -- Target forward ------------------------------------------------
        backend.set_placement(target_pl)
        target_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        target_logits = target_out.logits  # (B, max_len, V)

        # -- Draft forward -------------------------------------------------
        backend.set_placement(draft_pl)
        draft_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        draft_logits = draft_out.logits  # (B, max_len, V)

        # -- Per-sample metrics --------------------------------------------
        assert self._trace_writer is not None

        for i, (problem, trace) in enumerate(batch):
            p_len = prompt_lens[i]
            c_len = completion_lens[i]
            offset = pad_lens[i]

            # Logits at position j predict token j+1.
            start = offset + p_len - 1
            end = offset + p_len + c_len - 1

            t_probs = F.softmax(
                target_logits[i, start:end, :].float(), dim=-1
            )  # (C, V)
            d_probs = F.softmax(
                draft_logits[i, start:end, :].float(), dim=-1
            )  # (C, V)

            # ── Token acceptance probability ──────────────────────────
            alpha = torch.min(t_probs, d_probs).sum(dim=-1)  # (C,)
            alpha_list = alpha.tolist()

            # ── Step acceptance rate (IS-estimated) ───────────────────
            comp_ids = torch.tensor(
                trace.completion_tokens, dtype=torch.long, device=device
            )
            p_x = t_probs[torch.arange(c_len, device=device), comp_ids]
            q_x = d_probs[torch.arange(c_len, device=device), comp_ids]

            block_rates = _compute_step_acceptance_rates(
                p_x, q_x, gamma
            )

            mean_step_rate = (
                sum(block_rates) / len(block_rates)
                if block_rates
                else 0.0
            )

            record = AcceptanceRateRecord(
                problem_id=trace.problem_id,
                task_name=trace.task_name,
                sample_idx=trace.sample_idx,
                num_completion_tokens=c_len,
                mean_token_acceptance_prob=float(alpha.mean()),
                per_token_acceptance_probs=tuple(alpha_list),
                gamma=gamma,
                mean_step_acceptance_rate=mean_step_rate,
                step_acceptance_rates=tuple(block_rates),
                draft_placement=draft_pl,
                target_placement=target_pl,
            )
            self._trace_writer.append([record])
            progress.advance(progress_task)


def _compute_step_acceptance_rates(
    p_x: torch.Tensor,
    q_x: torch.Tensor,
    gamma: int,
) -> list[float]:
    """Compute IS-estimated step acceptance rate for each block of gamma tokens.

    For each block of gamma tokens x_{1:gamma} (generated from target p):

    1. Per-token acceptance probability: ``min(1, p(x_i)/q(x_i))``
    2. Expected accepted count (sequential): ``sum_k prod_{j=1}^{k} accept_prob_j``
    3. IS weight (correcting p -> q): ``prod q(x_i)/p(x_i)``
    4. IS-corrected step rate: ``is_weight * expected_accepted / gamma``

    Args:
        p_x: Target probabilities for the actual tokens, shape ``(C,)``.
        q_x: Draft probabilities for the actual tokens, shape ``(C,)``.
        gamma: Draft length (block size).

    Returns:
        List of per-block step acceptance rates.
    """
    c_len = p_x.shape[0]
    if c_len == 0:
        return []

    eps = 1e-30
    p_safe = p_x.clamp(min=eps)
    q_safe = q_x.clamp(min=eps)

    # min(1, p/q) per token
    accept_probs = torch.clamp(p_safe / q_safe, max=1.0)
    # log(q/p) per token for IS weights
    log_is = torch.log(q_safe) - torch.log(p_safe)

    block_rates: list[float] = []
    for s in range(0, c_len, gamma):
        e = min(s + gamma, c_len)
        actual_g = e - s

        block_accept = accept_probs[s:e]  # (g,)
        block_log_is = log_is[s:e]  # (g,)

        # expected accepted = sum_{k=1}^{g} prod_{j=1}^{k} accept_prob_j
        cum_prod = torch.cumprod(block_accept, dim=0)
        expected_accepted = cum_prod.sum().item()

        log_is_weight = block_log_is.sum().item()
        log_is_weight = max(min(log_is_weight, 50.0), -50.0)
        is_weight = math.exp(log_is_weight)

        block_rate = is_weight * expected_accepted / actual_g
        block_rates.append(block_rate)

    return block_rates


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class AcceptanceRateEvaluatorConfig(TraceBasedEvaluatorConfig):
    """Configuration for acceptance rate estimation evaluator."""

    _impl_class: ClassVar[type[BaseEvaluator]] = AcceptanceRateEvaluator

    evaluator_type: Literal["acceptance_rate"] = "acceptance_rate"

    target_placement: list[str] = Field(
        description="Target placement pattern (cyclically extended to num_layers). "
        "Defaults to ['attention'] (all-attention).",
    )

    gamma: int = Field(
        default=8,
        gt=0,
        description="Draft length (block size) for step acceptance rate estimation.",
    )
