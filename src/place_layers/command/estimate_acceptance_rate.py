"""Estimate speculative decoding acceptance rates between two placements.

Computes two complementary metrics from pre-computed teacher traces:

1. **Token acceptance probability** (alpha): the analytical per-position
   probability that a draft token is accepted by rejection sampling::

       alpha_i = sum_v min(p_target(v | x_{<i}), p_draft(v | x_{<i}))
               = 1 - D_TV(p_target, p_draft)

2. **Step acceptance rate**: the expected fraction of draft tokens accepted
   per speculative decoding step of length gamma, estimated via importance
   sampling.

Requires ``TransformersBackend`` for direct model access to obtain
full-vocabulary logit tensors.

Usage
-----
    python -m place_layers \\
        +command=estimate_acceptance_rate \\
        +backend=transformers \\
        +model=apriel2_0_5b_dev \\
        +placement=every2nd-gdn
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, ClassVar, Literal

import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from pydantic import Field

from place_layers.backend import BackendConfig
from place_layers.backend.transformers import TransformersBackend
from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.evaluator.progress import create_progress
from place_layers.evaluator.traces.types import (
    AcceptanceRateRecord,
    Trace,
    TraceDataset,
    TraceRecordAdapter,
)

logger = logging.getLogger(__name__)


class EstimateAcceptanceRateCommand(BaseCommand):
    """Estimate acceptance rates between target and draft placements."""

    config: "EstimateAcceptanceRateCommandConfig"

    async def run(self) -> None:
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)
        logger.info(f"Output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load traces from all sources
        all_traces = self._load_traces()
        if not all_traces:
            logger.warning("No traces found in trace_sources")
            return

        logger.info(f"Loaded {len(all_traces)} traces")

        # Set up backend
        async with self.config.backend.create_backend() as backend:
            if not isinstance(backend, TransformersBackend):
                raise TypeError(
                    f"estimate_acceptance_rate requires TransformersBackend, "
                    f"got {type(backend).__name__}. "
                    f"Full-vocabulary logit computation needs direct model "
                    f"access."
                )

            num_layers = backend.model_structure.num_layers
            gamma = self.config.gamma

            # Resolve placements
            target_pattern = self.config.target_placement
            target_pl = [
                target_pattern[i % len(target_pattern)]
                for i in range(num_layers)
            ]
            draft_pattern = self.config.placement or target_pattern
            draft_pl = [
                draft_pattern[i % len(draft_pattern)]
                for i in range(num_layers)
            ]

            logger.info(f"Target placement: {target_pl}")
            logger.info(f"Draft placement:  {draft_pl}")
            logger.info(f"Gamma: {gamma}")

            # Check for already-completed traces (resumption)
            results_path = output_dir / "results.jsonl"
            completed = self._load_completed(results_path)
            remaining = [
                t for t in all_traces
                if (t.problem_id, t.sample_idx) not in completed
            ]

            if not remaining:
                logger.info("All traces already processed")
            else:
                logger.info(
                    f"Processing {len(remaining)} traces "
                    f"({len(completed)} already done) "
                    f"in batches of {backend.config.batch_size}"
                )

                batch_size = backend.config.batch_size
                with create_progress() as progress:
                    ptask = progress.add_task(
                        "[cyan]acceptance rate", total=len(remaining)
                    )
                    for start in range(0, len(remaining), batch_size):
                        batch = remaining[start : start + batch_size]
                        records = _process_batch(
                            batch, target_pl, draft_pl,
                            backend, gamma,
                        )
                        _append_records(results_path, records)
                        progress.advance(ptask, len(batch))

        # Summarize
        summary = self._summarize(results_path, output_dir)
        logger.info(
            f"Done. token_acceptance_prob={summary['overall_token_acceptance_prob']:.4f}, "
            f"step_acceptance_rate={summary['overall_step_acceptance_rate']:.4f}"
        )

    def _load_traces(self) -> list[Trace]:
        """Load Trace records from all configured trace source directories."""
        traces: list[Trace] = []
        for source in self.config.trace_sources:
            source_dir = Path(source)
            if not source_dir.exists():
                logger.warning(f"Trace source not found: {source_dir}")
                continue
            dataset = TraceDataset.load_from_directory(source_dir)
            traces.extend(dataset.traces)
            logger.info(
                f"  Loaded {len(dataset.traces)} traces from {source_dir}"
            )
        return traces

    def _load_completed(
        self, results_path: Path
    ) -> set[tuple[str, int]]:
        """Load already-completed (problem_id, sample_idx) pairs for resumption."""
        completed: set[tuple[str, int]] = set()
        if not results_path.exists():
            return completed
        for line in results_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = TraceRecordAdapter.validate_json(line)
            if isinstance(rec, AcceptanceRateRecord):
                completed.add((rec.problem_id, rec.sample_idx))
        return completed

    def _summarize(
        self, results_path: Path, output_dir: Path
    ) -> dict[str, Any]:
        """Aggregate AcceptanceRateRecords into a summary JSON."""
        records: list[AcceptanceRateRecord] = []
        if results_path.exists():
            for line in results_path.read_text().splitlines():
                if not line.strip():
                    continue
                rec = TraceRecordAdapter.validate_json(line)
                if isinstance(rec, AcceptanceRateRecord):
                    records.append(rec)

        if not records:
            logger.warning("No AcceptanceRateRecords found")
            summary: dict[str, Any] = {
                "overall_token_acceptance_prob": 0.0,
                "overall_step_acceptance_rate": 0.0,
                "gamma": self.config.gamma,
                "task_results": {},
            }
            (output_dir / "acceptance_rate_summary.json").write_text(
                json.dumps(summary, indent=2)
            )
            return summary

        by_task: dict[str, list[AcceptanceRateRecord]] = {}
        for rec in records:
            by_task.setdefault(rec.task_name, []).append(rec)

        task_results: dict[str, dict[str, Any]] = {}
        for task_name, task_records in sorted(by_task.items()):
            n = len(task_records)
            token_mean = sum(
                r.mean_token_acceptance_prob for r in task_records
            ) / n
            step_mean = sum(
                r.mean_step_acceptance_rate for r in task_records
            ) / n
            task_results[task_name] = {
                "mean_token_acceptance_prob": round(token_mean, 6),
                "mean_step_acceptance_rate": round(step_mean, 6),
                "num_samples": n,
            }
            logger.info(
                f"  {task_name}: token_acceptance_prob={token_mean:.4f}, "
                f"step_acceptance_rate={step_mean:.4f} "
                f"({n} samples)"
            )

        n_total = len(records)
        overall_token = sum(
            r.mean_token_acceptance_prob for r in records
        ) / n_total
        overall_step = sum(
            r.mean_step_acceptance_rate for r in records
        ) / n_total

        summary = {
            "overall_token_acceptance_prob": round(overall_token, 6),
            "overall_step_acceptance_rate": round(overall_step, 6),
            "gamma": self.config.gamma,
            "num_samples": n_total,
            "task_results": task_results,
        }
        summary_path = output_dir / "acceptance_rate_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        logger.info(f"Summary written to {summary_path}")
        return summary


# ---------------------------------------------------------------------------
# Batch processing (standalone, no class state needed)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _process_batch(
    batch: list[Trace],
    target_pl: list[str],
    draft_pl: list[str],
    backend: TransformersBackend,
    gamma: int,
) -> list[AcceptanceRateRecord]:
    """Run target + draft forward passes and compute both metrics."""
    device = backend.device
    model = backend.model

    sequences: list[list[int]] = []
    prompt_lens: list[int] = []
    completion_lens: list[int] = []

    for trace in batch:
        seq = list(trace.prompt_tokens) + list(trace.completion_tokens)
        sequences.append(seq)
        prompt_lens.append(len(trace.prompt_tokens))
        completion_lens.append(len(trace.completion_tokens))

    max_len = max(len(s) for s in sequences)
    pad_id = getattr(backend.tokenizer, "pad_token_id", None) or 0

    # Right-pad so recurrent layers (GDN/KDA) don't process padding
    # tokens before the real sequence.
    padded = []
    attention_masks = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        padded.append(seq + [pad_id] * pad_len)
        attention_masks.append([1] * len(seq) + [0] * pad_len)

    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    attention_mask = torch.tensor(
        attention_masks, dtype=torch.long, device=device
    )

    # Target forward
    backend.set_placement(target_pl)
    target_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits

    # Draft forward
    backend.set_placement(draft_pl)
    draft_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits

    # Per-sample metrics
    records: list[AcceptanceRateRecord] = []
    for i, trace in enumerate(batch):
        p_len = prompt_lens[i]
        c_len = completion_lens[i]

        start = p_len - 1
        end = p_len + c_len - 1

        t_probs = F.softmax(
            target_logits[i, start:end, :].float(), dim=-1
        )
        d_probs = F.softmax(
            draft_logits[i, start:end, :].float(), dim=-1
        )

        # Token acceptance probability
        alpha = torch.min(t_probs, d_probs).sum(dim=-1)

        # Step acceptance rate (from per-position alphas)
        step_rates = _compute_step_acceptance_rates(alpha, gamma)
        mean_step = (
            sum(step_rates) / len(step_rates)
            if step_rates else 0.0
        )

        records.append(AcceptanceRateRecord(
            problem_id=trace.problem_id,
            task_name=trace.task_name,
            sample_idx=trace.sample_idx,
            num_completion_tokens=c_len,
            mean_token_acceptance_prob=float(alpha.mean()),
            per_token_acceptance_probs=tuple(alpha.tolist()),
            gamma=gamma,
            mean_step_acceptance_rate=mean_step,
            step_acceptance_rates=tuple(step_rates),
            draft_placement=draft_pl,
            target_placement=target_pl,
        ))

    return records


def _append_records(
    path: Path, records: list[AcceptanceRateRecord]
) -> None:
    """Append records to a JSONL file."""
    with path.open("a") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")


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

    where ``alpha_j = sum_v min(p(v), q(v))`` is the exact marginal
    acceptance probability at position j.  Assumes acceptance decisions
    are independent given the context. Note that this estimation is approximate;
    within each step the tokens are generated according to the target rather than
    the draft as in speculative decoding. We do not use importance weights due
    to high variance.

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



# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class EstimateAcceptanceRateCommandConfig(BaseCommandConfig):
    """Configuration for acceptance rate estimation command."""

    _impl_class: ClassVar[type[BaseCommand]] = EstimateAcceptanceRateCommand

    command: Literal["estimate_acceptance_rate"] = "estimate_acceptance_rate"

    backend: BackendConfig = Field(
        description="Backend configuration (must be TransformersBackend).",
    )

    placement: list[str] | None = Field(
        default=None,
        description="Draft placement (mixer names per layer, cyclically extended). "
        "If null, defaults to target_placement.",
    )

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
