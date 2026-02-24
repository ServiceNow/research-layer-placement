"""Speculative decoding evaluator for supernet placement evaluation.

This evaluator uses speculative decoding (Leviathan et al., 2023) to measure
task accuracy. A "draft" placement proposes tokens cheaply, and a "target"
placement verifies them via rejection sampling. The output distribution is
**exactly** that of the target model.

In addition to standard accuracy metrics, this evaluator tracks speculative
decoding efficiency: acceptance rate, target model calls, and tokens generated.

Algorithm (one speculative step):
    Given context x_{<t} and draft length gamma:
    1. Draft gamma tokens from q (draft placement) autoregressively.
    2. Score all gamma tokens under p (target placement) in one forward pass.
    3. Accept/reject left-to-right: accept x_i with prob min(1, p(x_i)/q(x_i)).
       On first rejection at position n+1, sample from norm(max(0, p - q)).
    4. If all accepted, sample bonus token from p at position gamma+1.

Key constraint: Only works with TransformersBackend (requires direct model access).

Usage
-----
    config = SpeculativeEvaluatorConfig(
        backend=TransformersBackendConfig(model_path="/path/to/model", ...),
        tasks=["gsm8k"],
        gamma=8,
        target_placement=["attention"],
        ...
    )
    evaluator = config.create_evaluator()
    score = await evaluator.evaluate(placement)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Sequence

import torch
import torch.nn.functional as F
from pydantic import Field, model_validator

from place_layers.backend.transformers import TransformersBackend
from place_layers.clients.types import GenerationResult
from place_layers.core.memory_logging import log_memory
from place_layers.core.types import LayerPlacement
from place_layers.evaluator.base_evaluator import BaseEvaluator
from place_layers.evaluator.base_generative_evaluator import (
    BaseGenerativeEvaluator,
    BaseGenerativeEvaluatorConfig,
)
from place_layers.evaluator.lm_eval_utils import (
    build_chat_messages,
    build_text_prompt,
    check_correctness,
)
from place_layers.evaluator.progress import create_progress
from place_layers.evaluator.traces.types import (
    Generation,
    SkippedProblem,
    SpeculativeGeneration,
    TraceRecord,
)
from place_layers.evaluator.traces.writer import summarize_generations

if TYPE_CHECKING:
    from lm_eval.api.task import Task
    from transformers import PreTrainedModel

    from place_layers.evaluator.traces.writer import GenerationSummary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures for per-step and per-problem results
# ---------------------------------------------------------------------------


@dataclass
class _StepResult:
    """Result of one speculative decoding step."""

    gamma: int  # draft tokens attempted
    num_accepted: int  # accepted before first rejection
    all_accepted: bool  # True if all gamma accepted
    tokens_emitted: list[int]  # accepted draft tokens + 1 new token


@dataclass
class _ProblemResult:
    """Speculative decoding results for one problem."""

    problem_idx: int
    steps: list[_StepResult] = field(default_factory=list)

    @property
    def total_draft_tokens(self) -> int:
        return sum(s.gamma for s in self.steps)

    @property
    def total_accepted(self) -> int:
        return sum(s.num_accepted for s in self.steps)

    @property
    def total_emitted(self) -> int:
        return sum(len(s.tokens_emitted) for s in self.steps)

    @property
    def num_target_calls(self) -> int:
        """Number of target model forward passes (= number of speculative steps)."""
        return len(self.steps)

    @property
    def step_acceptance_rate(self) -> float:
        total = self.total_draft_tokens
        return self.total_accepted / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class SpeculativeEvaluator(BaseGenerativeEvaluator):
    """Evaluator that uses speculative decoding with draft/target placements.

    The draft placement (specified via ``+placement=...``) proposes tokens
    cheaply, and the target placement (``evaluator.target_placement``,
    default ``[attention]``) verifies them.  The output distribution is
    guaranteed to match the target model exactly.

    Only works with ``TransformersBackend`` (requires direct ``model.generate``
    and ``model(input_ids)`` access).

    Must be used as an async context manager:

        async with config.create_evaluator() as evaluator:
            score = await evaluator.evaluate(placement)
    """

    config: "SpeculativeEvaluatorConfig"

    def __init__(
        self, config: "SpeculativeEvaluatorConfig", *, output_dir: Path
    ) -> None:
        super().__init__(config, output_dir=output_dir)
        self._speculative_metrics: dict[str, float] | None = None

    # ------------------------------------------------------------------
    # Abstract contract from BaseGenerativeEvaluator
    # ------------------------------------------------------------------

    def _make_records(
        self, task: "Task", doc: dict, idx: int, result: GenerationResult
    ) -> Sequence[TraceRecord]:
        """Create Generation records (satisfies abstract contract).

        Not called through the speculative evaluate() path, but provides a
        valid implementation for the interface.
        """
        task_name = task.config.task or ""
        problem_id = f"{task_name}:{idx}"
        records: list[Generation] = []
        for sample_idx, sample in enumerate(result.samples):
            correct = check_correctness(task, doc, sample.text)
            records.append(
                Generation(
                    prompt_tokens=tuple(result.prompt_token_ids),
                    completion_tokens=tuple(sample.token_ids),
                    correct=correct,
                    truncated=sample.truncated,
                    problem_id=problem_id,
                    task_name=task_name,
                    sample_idx=sample_idx,
                    completion_text=sample.text,
                )
            )
        return records

    # ------------------------------------------------------------------
    # Extra metrics interface
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict[str, float] | None:
        """Return speculative decoding metrics collected during evaluation."""
        return self._speculative_metrics

    # ------------------------------------------------------------------
    # Logits warper construction (mirrors HF generate internals)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_warpers(
        temperature: float, top_p: float
    ) -> "torch.nn.Module | None":
        """Build the same logits warpers that ``model.generate()`` applies.

        Returns a ``LogitsProcessorList`` or *None* when no warping is needed
        (temperature == 1.0 and top_p == 1.0).
        """
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

    # ------------------------------------------------------------------
    # KV cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_empty_cache(model: "PreTrainedModel") -> Any:
        """Create an empty KV cache suitable for the model.

        Uses the model's ``_prepare_cache_for_generation`` hook so we
        stay backend-agnostic (works for Apriel2Cache or any HF cache).
        """
        gen_config = copy.copy(model.generation_config)
        gen_config.use_cache = True
        model_kwargs: dict[str, Any] = {}
        model._prepare_cache_for_generation(
            gen_config, model_kwargs, None, 1, None
        )
        cache = model_kwargs.get("past_key_values")
        if cache is None:
            raise RuntimeError(
                "Model did not create a cache via _prepare_cache_for_generation. "
                "KV cache for speculative decoding requires a model that "
                "supports caching."
            )
        return cache

    @staticmethod
    def _clone_cache(cache: Any) -> Any:
        """Deep-copy a cache so the original is not mutated.

        All tensors inside the cache are cloned (PyTorch's
        ``__deepcopy__`` calls ``.clone().detach()``).
        """
        return copy.deepcopy(cache)

    @staticmethod
    def _validate_target_all_attention(
        target_pl: list[str], model: "PreTrainedModel"
    ) -> None:
        """Validate that every layer in *target_pl* is an attention mixer.

        KV-cached speculative decoding requires the target to be
        all-attention because backtracking (``cache.crop``) only works
        for attention caches (which have a sequence dimension).

        Raises ``ValueError`` if any layer is non-attention.
        """
        gen_config = copy.copy(model.generation_config)
        gen_config.use_cache = True
        model_kwargs: dict[str, Any] = {}
        model._prepare_cache_for_generation(
            gen_config, model_kwargs, None, 1, None
        )
        cache = model_kwargs.get("past_key_values")
        if cache is None:
            raise RuntimeError(
                "Model did not create a cache; cannot validate target "
                "placement for KV-cached speculative decoding."
            )

        for i, mixer_name in enumerate(target_pl):
            layer = cache.layers[i]
            if isinstance(layer, dict):
                # Stochastic layer — check the specific sub-mixer
                sub = layer.get(mixer_name)
                if sub is None:
                    raise ValueError(
                        f"Mixer '{mixer_name}' not found in stochastic "
                        f"layer {i}. Available: {list(layer.keys())}"
                    )
                if not hasattr(sub, "cumulative_length"):
                    raise ValueError(
                        f"KV cache for speculative decoding requires "
                        f"target_placement to be all-attention. Layer {i} "
                        f"mixer '{mixer_name}' is not an attention mixer."
                    )
            else:
                # Non-stochastic layer
                if not hasattr(layer, "cumulative_length"):
                    raise ValueError(
                        f"KV cache for speculative decoding requires "
                        f"target_placement to be all-attention. Layer {i} "
                        f"is not an attention layer."
                    )

        del cache

    # ------------------------------------------------------------------
    # Core speculative decoding
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _speculative_decode_step(
        self,
        context_ids: torch.Tensor,  # (1, context_len)
        gamma: int,
        draft_pl: list[str],
        target_pl: list[str],
        backend: TransformersBackend,
        warpers: "torch.nn.Module | None",
        draft_cache: Any,
        target_cache: Any,
        next_draft_logits: torch.Tensor,  # (1, vocab_size)
    ) -> tuple[_StepResult, torch.Tensor]:
        """Execute one speculative decoding step.

        Uses a custom draft loop (not ``model.generate``) so we control
        the KV cache precisely.  The target forward only processes tokens
        not yet in ``target_cache``.

        Cache contract
        --------------
        * ``draft_cache`` is mutated in place: extended with KV/state for
          all emitted tokens (via the refresh forward at the end).
        * ``target_cache`` is mutated in place: extended by the target
          forward, then ``crop``-ped to ``accepted_length`` on rejection.

        Args:
            context_ids: Current token sequence ``(1, context_len)``.
            gamma: Number of draft tokens to propose.
            draft_pl, target_pl: Full placements.
            backend: TransformersBackend instance.
            warpers: LogitsProcessorList for temperature/top_p (or None).
            draft_cache: Cache covering ``context_ids`` (mutated).
            target_cache: Target cache (mutated).
            next_draft_logits: Logits for the first draft-token position,
                from the previous refresh/prefill forward.

        Returns:
            ``(_StepResult, next_draft_logits_for_next_step)``
        """
        model = backend.model
        tokenizer = backend.tokenizer
        device = backend.device
        context_len = context_ids.shape[1]

        # ── Step 1: Custom draft loop ──────────────────────────────────
        backend.set_placement(draft_pl)
        draft_cache_copy = self._clone_cache(draft_cache)

        draft_ids: list[int] = []
        q_probs_list: list[torch.Tensor] = []
        current_logits = next_draft_logits  # (1, vocab_size)

        for i in range(gamma):
            # Apply warpers to match the sampling distribution
            if warpers is not None:
                dummy = torch.zeros(1, 1, dtype=torch.long, device=device)
                warped = warpers(dummy, current_logits.clone())
            else:
                warped = current_logits
            probs = F.softmax(warped, dim=-1)  # (1, vocab)
            q_probs_list.append(probs[0])  # (vocab,)

            # Sample next draft token
            token_id = torch.multinomial(probs[0], 1).item()
            draft_ids.append(token_id)

            if token_id == tokenizer.eos_token_id:
                break

            # Forward for next position (skip for last draft token)
            if i < gamma - 1:
                token_t = torch.tensor([[token_id]], device=device)
                out = model(
                    input_ids=token_t,
                    past_key_values=draft_cache_copy,
                    use_cache=True,
                )
                current_logits = out.logits[:, -1, :]  # (1, vocab)

        actual_gamma = len(draft_ids)
        if actual_gamma == 0:
            return _StepResult(0, 0, False, []), next_draft_logits

        q_probs = torch.stack(q_probs_list)  # (actual_gamma, vocab)

        # ── Step 2: Target scoring with cache ──────────────────────────
        backend.set_placement(target_pl)

        # Build full sequence and pass only new tokens to target
        draft_tensor = torch.tensor([draft_ids], device=device)
        full_sequence = torch.cat([context_ids, draft_tensor], dim=1)

        target_cache_len = target_cache.get_seq_length()
        target_input = full_sequence[:, target_cache_len:]

        target_out = model(
            input_ids=target_input,
            past_key_values=target_cache,
            use_cache=True,
        )
        # target_cache now covers full_sequence

        # Extract gamma+1 logits for accept/reject.
        # Draft tokens start at (context_len - target_cache_len) in
        # target_input; we need one logit *before* them.
        draft_start_in_input = context_len - target_cache_len
        raw_logits = target_out.logits[
            0, draft_start_in_input - 1 : draft_start_in_input + actual_gamma
        ]  # (actual_gamma + 1, vocab)

        if warpers is not None:
            dummy_ids = torch.zeros(
                raw_logits.shape[0], 1, dtype=torch.long, device=device
            )
            adjusted = warpers(dummy_ids, raw_logits.clone())
        else:
            adjusted = raw_logits
        p_probs = F.softmax(adjusted, dim=-1)

        # ── Step 3: Accept / reject ────────────────────────────────────
        n_accepted = 0

        for i in range(actual_gamma):
            x_i = draft_ids[i]
            p_x = p_probs[i, x_i].item()
            q_x = q_probs[i, x_i].item()
            accept_prob = min(1.0, p_x / q_x) if q_x > 0 else 1.0

            if torch.rand(1, device=device).item() < accept_prob:
                n_accepted += 1
            else:
                # Rejected — sample from residual distribution
                residual = torch.clamp(p_probs[i] - q_probs[i], min=0)
                residual_sum = residual.sum()
                if residual_sum > 0:
                    replacement = torch.multinomial(
                        residual / residual_sum, 1
                    ).item()
                else:
                    replacement = torch.multinomial(p_probs[i], 1).item()

                emitted = draft_ids[:n_accepted] + [replacement]

                # ── Target backtrack: crop to accepted length ──────────
                accepted_length = context_len + n_accepted
                target_cache.crop(accepted_length)

                # ── Draft refresh: process emitted tokens ──────────────
                backend.set_placement(draft_pl)
                emitted_t = torch.tensor([emitted], device=device)
                refresh_out = model(
                    input_ids=emitted_t,
                    past_key_values=draft_cache,
                    use_cache=True,
                )
                new_logits = refresh_out.logits[:, -1, :]  # (1, vocab)

                return (
                    _StepResult(actual_gamma, n_accepted, False, emitted),
                    new_logits,
                )

        # All accepted — sample bonus token from p at position gamma+1
        bonus = torch.multinomial(p_probs[actual_gamma], 1).item()
        emitted = draft_ids + [bonus]

        # Target cache already covers context + all draft tokens.
        # The bonus token is NOT in the target cache; it will be
        # picked up by the next step's target forward.

        # ── Draft refresh: process emitted tokens ──────────────────────
        backend.set_placement(draft_pl)
        emitted_t = torch.tensor([emitted], device=device)
        refresh_out = model(
            input_ids=emitted_t,
            past_key_values=draft_cache,
            use_cache=True,
        )
        new_logits = refresh_out.logits[:, -1, :]  # (1, vocab)

        return (
            _StepResult(actual_gamma, actual_gamma, True, emitted),
            new_logits,
        )

    @torch.inference_mode()
    def _run_speculative_problem(
        self,
        prompt_ids: torch.Tensor,
        draft_pl: list[str],
        target_pl: list[str],
        backend: TransformersBackend,
        warpers: "torch.nn.Module | None",
        problem_idx: int,
    ) -> _ProblemResult:
        """Run speculative decoding for a single problem until completion.

        Creates per-problem ``draft_cache`` and ``target_cache``, does a
        draft prefill on the prompt, then loops with
        ``_speculative_decode_step``.

        Args:
            prompt_ids: Tokenized prompt ``(1, prompt_len)``.
            draft_pl: Full draft placement.
            target_pl: Full target placement.
            backend: TransformersBackend instance.
            warpers: Logits warpers (or None).
            problem_idx: Index for tracking.

        Returns:
            ``_ProblemResult`` with all step results.
        """
        model = backend.model
        device = backend.device
        context_ids = prompt_ids.clone()
        result = _ProblemResult(problem_idx=problem_idx)
        tokens_generated = 0
        eos_id = backend.tokenizer.eos_token_id

        # ── Create caches ──────────────────────────────────────────────
        draft_cache = self._create_empty_cache(model)
        target_cache = self._create_empty_cache(model)

        # ── Draft prefill: fill draft_cache with prompt KV ─────────────
        backend.set_placement(draft_pl)
        prefill_out = model(
            input_ids=context_ids,
            past_key_values=draft_cache,
            use_cache=True,
        )
        next_draft_logits = prefill_out.logits[:, -1, :]  # (1, vocab)

        # ── Speculative decoding loop ──────────────────────────────────
        while tokens_generated < self.config.max_new_tokens:
            g = min(
                self.config.gamma,
                self.config.max_new_tokens - tokens_generated,
            )
            step, next_draft_logits = self._speculative_decode_step(
                context_ids,
                g,
                draft_pl,
                target_pl,
                backend,
                warpers,
                draft_cache,
                target_cache,
                next_draft_logits,
            )

            if not step.tokens_emitted:
                break

            result.steps.append(step)
            new_tokens = torch.tensor(
                [step.tokens_emitted], device=device
            )
            context_ids = torch.cat([context_ids, new_tokens], dim=1)
            tokens_generated += len(step.tokens_emitted)

            if eos_id is not None and eos_id in step.tokens_emitted:
                break

        return result

    # ------------------------------------------------------------------
    # Prompt tokenization
    # ------------------------------------------------------------------

    def _tokenize_prompt(
        self,
        task: "Task",
        doc: dict,
        backend: TransformersBackend,
    ) -> torch.Tensor:
        """Tokenize a problem prompt using the backend's tokenizer.

        Returns:
            Token IDs tensor of shape ``(1, prompt_len)`` on device.
        """
        tokenizer = backend.tokenizer
        if self.config.use_chat_template:
            messages = build_chat_messages(
                task, doc, task.config.num_fewshot or 0
            )
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = build_text_prompt(
                task, doc, task.config.num_fewshot or 0
            )
        encoding = tokenizer(prompt_text, return_tensors="pt")
        return encoding.input_ids.to(backend.device)

    # ------------------------------------------------------------------
    # Main evaluation interface (overrides BaseGenerativeEvaluator)
    # ------------------------------------------------------------------

    async def evaluate(self, parameters: LayerPlacement | None) -> float:
        """Evaluate using speculative decoding with draft/target placements.

        The *parameters* argument is the **draft** placement (from ``+placement=``).
        The target placement comes from ``self.config.target_placement``.

        Args:
            parameters: Draft layer placement, or None for default.

        Returns:
            Problem accuracy (fraction with any correct sample).
        """
        # Validate backend type
        if not isinstance(self._backend, TransformersBackend):
            raise TypeError(
                f"SpeculativeEvaluator requires TransformersBackend, "
                f"got {type(self._backend).__name__}. "
                f"Speculative decoding needs direct model access."
            )
        backend: TransformersBackend = self._backend

        # Set draft placement (also used for trace writer subdirectory)
        if parameters is not None:
            await self._set_placement(parameters)
        self._init_trace_writer()
        assert self._trace_writer is not None

        # Resolve target placement by cyclic extension
        target_pattern = self.config.target_placement
        target_pl = [
            target_pattern[i % len(target_pattern)] for i in range(self.num_layers)
        ]

        # Resolve draft placement (full-length from _current_placement)
        draft_pl = list(self._current_placement) if self._current_placement else target_pl

        # Validate target placement mixer names
        valid_names = set(self.mixer_names)
        for name in target_pl:
            if name not in valid_names:
                raise ValueError(
                    f"Invalid mixer name '{name}' in target_placement. "
                    f"Valid names: {sorted(valid_names)}"
                )

        # Validate target is all-attention (required for cache backtracking)
        self._validate_target_all_attention(target_pl, backend.model)

        logger.info(
            f"Speculative decoding: gamma={self.config.gamma}, "
            f"draft={draft_pl[:4]}..., target={target_pl[:4]}..."
        )

        # Build logits warpers
        warpers = self._make_warpers(self.config.temperature, self.config.top_p)

        log_memory("SPECULATIVE EVAL START")

        # Collect per-problem stats for aggregate metrics
        problem_results: list[_ProblemResult] = []
        processed = 0

        with create_progress() as progress:
            progress_task = progress.add_task(
                "[cyan]Speculative decoding", total=None
            )

            for task_spec in self.config.tasks:
                prepared = await self._prepare_task(task_spec, "generate_until")
                if prepared is None:
                    continue

                task, task_name, docs, remaining = prepared
                logger.info(
                    f"Task '{task_name}': evaluating {len(remaining)} problems "
                    f"via speculative decoding"
                )

                for idx in remaining:
                    doc = docs[idx]
                    problem_id = f"{task_name}:{idx}"

                    # Tokenize prompt
                    prompt_ids = self._tokenize_prompt(task, doc, backend)

                    # Run speculative decoding
                    prob_result = self._run_speculative_problem(
                        prompt_ids, draft_pl, target_pl, backend, warpers, idx
                    )
                    problem_results.append(prob_result)

                    # Decode completion and check correctness
                    all_emitted: list[int] = []
                    for step in prob_result.steps:
                        all_emitted.extend(step.tokens_emitted)

                    # Trim at EOS if present
                    eos_id = backend.tokenizer.eos_token_id
                    if eos_id is not None and eos_id in all_emitted:
                        eos_pos = all_emitted.index(eos_id)
                        all_emitted = all_emitted[:eos_pos]

                    completion_text = backend.tokenizer.decode(
                        all_emitted, skip_special_tokens=True
                    )

                    truncated = (
                        prob_result.total_emitted >= self.config.max_new_tokens
                    )
                    if truncated and not self.config.evaluate_truncated_samples:
                        correct = False
                    else:
                        correct = check_correctness(task, doc, completion_text)

                    # Write SpeculativeGeneration record
                    record = SpeculativeGeneration(
                        prompt_tokens=tuple(prompt_ids[0].tolist()),
                        completion_tokens=tuple(all_emitted),
                        correct=correct,
                        truncated=truncated,
                        problem_id=problem_id,
                        task_name=task_name,
                        sample_idx=0,
                        completion_text=completion_text,
                        num_speculative_steps=prob_result.num_target_calls,
                        total_draft_tokens=prob_result.total_draft_tokens,
                        num_accepted_tokens=prob_result.total_accepted,
                        step_acceptance_rate=prob_result.step_acceptance_rate,
                        draft_placement=draft_pl,
                        target_placement=target_pl,
                    )
                    self._trace_writer.append([record])
                    processed += 1
                    progress.advance(progress_task)

        log_memory("SPECULATIVE EVAL END")
        logger.info(f"Speculative evaluation complete: {processed} problems processed")

        # Compute accuracy summary from disk (single source of truth)
        summary = self._summarize()
        logger.info(f"\n{summary}")

        # Compute and store speculative decoding metrics
        if problem_results:
            avg_step_acceptance = sum(
                r.step_acceptance_rate for r in problem_results
            ) / len(problem_results)
            avg_target_calls = sum(
                r.num_target_calls for r in problem_results
            ) / len(problem_results)
            avg_tokens = sum(
                r.total_emitted for r in problem_results
            ) / len(problem_results)

            self._speculative_metrics = {
                "avg_step_acceptance_rate": round(avg_step_acceptance, 4),
                "avg_target_model_calls": round(avg_target_calls, 2),
                "avg_tokens_generated": round(avg_tokens, 2),
            }

            logger.info(
                f"Speculative metrics: "
                f"step_acceptance_rate={avg_step_acceptance:.4f}, "
                f"target_calls={avg_target_calls:.1f}, "
                f"tokens_generated={avg_tokens:.1f}"
            )
        else:
            self._speculative_metrics = None

        return summary.problem_accuracy

    def _summarize(self) -> "GenerationSummary":
        """Compute summary from disk (SpeculativeGeneration extends BaseGeneration)."""
        return summarize_generations(self._output_dir)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class SpeculativeEvaluatorConfig(BaseGenerativeEvaluatorConfig):
    """Configuration for speculative decoding evaluator.

    Extends ``BaseGenerativeEvaluatorConfig`` with speculative-specific
    parameters: *gamma* (draft tokens per step) and *target_placement*.
    """

    _impl_class: ClassVar[type[BaseEvaluator]] = SpeculativeEvaluator

    evaluator_type: Literal["speculative"] = "speculative"

    gamma: int = Field(
        description="Number of draft tokens to propose per speculative step",
        gt=0,
    )
    target_placement: list[str] = Field(
        description="Target model placement pattern (cyclically extended to num_layers). "
        "Defaults to [attention] (all-attention). Must be all-attention for "
        "KV cache backtracking. Draft placement may be hybrid (attention + SSM).",
    )

    @model_validator(mode="after")
    def validate_speculative_config(self) -> "SpeculativeEvaluatorConfig":
        """Validate speculative-specific constraints."""
        if not self.target_placement:
            raise ValueError("target_placement must be a non-empty list")
        if self.temperature <= 0:
            raise ValueError(
                "Speculative decoding requires temperature > 0 for valid "
                "rejection sampling. Use a small value like 0.01 for "
                "near-deterministic behavior."
            )
        return self
