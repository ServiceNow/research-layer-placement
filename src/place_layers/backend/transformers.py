"""TransformersBackend - Sync backend using HuggingFace Transformers."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, ClassVar, Literal, Self, Sequence

import torch
import torch.nn.functional as F
from pydantic import Field

from place_layers.backend.base import (
    BaseBackendConfig,
    SyncBackend,
    batched_expand_map,
    batched_map,
    batched_map_by_key,
)
from place_layers.backend.types import (
    ChatMessage,
    GenerateChatRequest,
    GenerateTextRequest,
    LogProbResult,
    ModelStructure,
    NoStructure,
    ScoreChatRequest,
    ScoreChatResult,
    ScoreTextRequest,
    ScoreTextResult,
    ScoreTokensRequest,
    ScoreTokensResult,
    SequenceScoringRequest,
    SupernetStructure,
)
from place_layers.clients.types import GeneratedSample, GenerationResult
from place_layers.core.memory_logging import (
    get_memory_stats,
    log_memory,
    log_memory_delta,
)

if TYPE_CHECKING:
    from place_layers.backend.apriel_validation import Apriel2Structure
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class TransformersBackend(SyncBackend):
    """Sync generation backend using HuggingFace Transformers.

    Handles model lifecycle, generation, and log probability computation.
    Supports dynamic mixer placement for Apriel2 supernet models.
    """

    def __init__(self, config: "TransformersBackendConfig") -> None:
        self._config = config
        self._model: "PreTrainedModel | None" = None
        self._tokenizer: "PreTrainedTokenizerBase | None" = None
        self._device: torch.device | None = None

        # Apriel2 supernet support
        self._apriel_structure: "Apriel2Structure | None" = None
        self._current_placement: list[str] | None = None

    @property
    def model(self) -> "PreTrainedModel":
        """Get the loaded model (raises if not in context)."""
        if self._model is None:
            raise RuntimeError("Backend not initialized. Use as context manager.")
        return self._model

    @property
    def tokenizer(self) -> "PreTrainedTokenizerBase":
        """Get the loaded tokenizer (raises if not in context)."""
        if self._tokenizer is None:
            raise RuntimeError("Backend not initialized. Use as context manager.")
        return self._tokenizer

    @property
    def device(self) -> torch.device:
        """Get the model's device (raises if not in context)."""
        if self._device is None:
            raise RuntimeError("Backend not initialized. Use as context manager.")
        return self._device

    def __enter__(self) -> Self:
        """Load model and tokenizer."""
        from transformers import AutoTokenizer

        from place_layers.backend.apriel_validation import (
            Apriel2ValidationError,
            validate_apriel2_model,
        )

        logger.info(f"Loading model from {self._config.model_path}...")
        log_memory("MODEL LOAD START")
        mem_before = get_memory_stats()
        start_time = time.perf_counter()

        # Load model
        self._model = self._load_model()
        self._model.eval()
        self._device = next(self._model.parameters()).device

        # Load tokenizer
        tokenizer_path = self._config.tokenizer_path or self._config.model_path
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=self._config.trust_remote_code,
            revision=self._config.revision,
        )

        # Ensure pad_token is set (required for batched generation)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        param_count = sum(p.numel() for p in self._model.parameters())
        elapsed = time.perf_counter() - start_time

        # Log memory after model load
        mem_after = get_memory_stats()
        logger.info(
            f"Model loaded in {elapsed:.1f}s: {param_count:,} params, "
            f"device={self._device}"
        )
        log_memory("MODEL LOAD END")
        log_memory_delta(mem_before, mem_after, "MODEL LOAD")

        # Validate Apriel2 supernet structure
        try:
            self._apriel_structure = validate_apriel2_model(self._model)
        except Apriel2ValidationError:
            # Not an Apriel2 supernet - leave structure as None
            pass

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Free GPU memory."""
        logger.info("Unloading model...")
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._device = None
        torch.cuda.empty_cache()

    def _load_model(self) -> "PreTrainedModel":
        """Load the model, detecting appropriate class from config."""
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
        )

        # Load config to determine model class from auto_map
        config = AutoConfig.from_pretrained(
            self._config.model_path,
            trust_remote_code=self._config.trust_remote_code,
            revision=self._config.revision,
        )
        auto_map = getattr(config, "auto_map", {})
        architectures = getattr(config, "architectures", [])

        if (
            "AutoModelForImageTextToText" in auto_map
            or "LlavaForConditionalGeneration" in architectures
        ):
            model_class = AutoModelForImageTextToText
        elif "AutoModelForCausalLM" in auto_map:
            model_class = AutoModelForCausalLM
        else:
            logger.info(
                f"No auto_map found (keys: {list(auto_map.keys())}), "
                f"defaulting to AutoModelForCausalLM"
            )
            model_class = AutoModelForCausalLM

        logger.info(f"Loading model with {model_class.__name__}")

        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self._config.torch_dtype, torch.bfloat16)

        # Build kwargs for model loading
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": self._config.device_map,
            "trust_remote_code": self._config.trust_remote_code,
            "revision": self._config.revision,
        }

        # Add attn_implementation if specified
        if self._config.attn_implementation is not None:
            load_kwargs["attn_implementation"] = self._config.attn_implementation
            logger.info(f"Using attention implementation: {self._config.attn_implementation}")

        model = model_class.from_pretrained(
            self._config.model_path,
            torch_dtype=torch_dtype,
            device_map=self._config.device_map,
            trust_remote_code=self._config.trust_remote_code,
            revision=self._config.revision,
        )
        return model

    @property
    def model_structure(self) -> ModelStructure:
        """Model structure (num_layers, mixer_names).

        Returns SupernetStructure for Apriel2 supernet models,
        NoStructure for non-supernet models.
        """
        if self._apriel_structure is None:
            return NoStructure()
        return SupernetStructure(
            num_layers=self._apriel_structure.num_layers,
            mixer_names=list(self._apriel_structure.mixer_names),
        )

    @property
    def placement(self) -> list[str] | None:
        """Current mixer placement, or None if model doesn't support placements."""
        if self._apriel_structure is None:
            return None
        # Read current placement from model
        return [
            self._model.model.decoder.blocks[i].mixer.main_mixer_name
            for i in range(self._apriel_structure.num_layers)
        ]

    # =========================================================================
    # Generation
    # =========================================================================

    # Key type for generation batching (all params that must match within a batch)
    _GenerateKey = tuple[int, int, float, float]  # (n, max_tokens, temperature, top_p)

    def generate_chat(
        self,
        requests: Iterable[GenerateChatRequest | None],
    ) -> Iterator[GenerationResult]:
        """Generate completions from chat messages, yielding as batches complete.

        Batching is handled internally based on batch_size config.
        Results are yielded in input order. Requests with different params
        trigger a batch flush.
        """

        def get_key(req: GenerateChatRequest) -> TransformersBackend._GenerateKey:
            return (req.n, req.max_tokens, req.temperature, req.top_p)

        def process_batch(
            key: TransformersBackend._GenerateKey, batch: list[GenerateChatRequest]
        ) -> list[GenerationResult]:
            n, max_tokens, temperature, top_p = key
            # Apply chat template to get prompt texts
            prompt_texts = []
            for req in batch:
                prompt_text = self.tokenizer.apply_chat_template(
                    list(req.messages),  # type: ignore[arg-type]
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt_texts.append(prompt_text)
            keys = [req.key for req in batch]
            return self._generate_from_texts(
                keys, prompt_texts, n, max_tokens, temperature, top_p
            )

        yield from batched_map_by_key(
            requests,
            get_key,
            process_batch,
            self._config.batch_size,
            self._config.flush_timeout,
        )

    def generate_text(
        self,
        requests: Iterable[GenerateTextRequest | None],
    ) -> Iterator[GenerationResult]:
        """Generate completions from text prompts, yielding as batches complete.

        Batching is handled internally based on batch_size config.
        Results are yielded in input order. Requests with different params
        trigger a batch flush.
        """

        def get_key(req: GenerateTextRequest) -> TransformersBackend._GenerateKey:
            return (req.n, req.max_tokens, req.temperature, req.top_p)

        def process_batch(
            key: TransformersBackend._GenerateKey, batch: list[GenerateTextRequest]
        ) -> list[GenerationResult]:
            n, max_tokens, temperature, top_p = key
            prompt_texts = [req.prompt for req in batch]
            keys = [req.key for req in batch]
            return self._generate_from_texts(
                keys, prompt_texts, n, max_tokens, temperature, top_p
            )

        yield from batched_map_by_key(
            requests,
            get_key,
            process_batch,
            self._config.batch_size,
            self._config.flush_timeout,
        )

    def _generate_from_texts(
        self,
        keys: list[str],
        prompt_texts: list[str],
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[GenerationResult]:
        """Generate n completions for a batch of text prompts (internal).

        Tokenizes all prompts together, runs a single batched generate call,
        and extracts log probabilities from generation scores.
        """
        if not prompt_texts:
            return []

        batch_size = len(prompt_texts)
        mem_before = get_memory_stats()
        logger.debug(
            f"_generate_from_texts: batch_size={batch_size}, n={n}, max_tokens={max_tokens}, "
            f"GPU alloc={mem_before.gpu_allocated_gb:.2f}GB"
        )

        # Tokenize all prompts with left padding (required for batched generation)
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        batch_encoding = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
        )

        self.tokenizer.padding_side = original_padding_side

        # Track original prompt lengths (before padding)
        prompt_lengths = [
            (batch_encoding.attention_mask[i] == 1).sum().item()
            for i in range(batch_size)
        ]

        # Move to device
        input_ids = batch_encoding.input_ids.to(self.device)
        attention_mask = batch_encoding.attention_mask.to(self.device)

        # Generate with scores for log probability extraction
        do_sample = temperature > 0
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p,
                do_sample=do_sample,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

        sequences = outputs.sequences
        scores = outputs.scores

        # Build results for each prompt
        results: list[GenerationResult] = []
        padded_prompt_len = input_ids.shape[1]

        for prompt_idx, key in enumerate(keys):
            prompt_start = padded_prompt_len - prompt_lengths[prompt_idx]
            prompt_tokens = input_ids[prompt_idx, prompt_start:].tolist()

            samples: list[GeneratedSample] = []

            for sample_idx in range(n):
                output_idx = prompt_idx * n + sample_idx
                full_sequence = sequences[output_idx].tolist()
                completion_tokens = full_sequence[padded_prompt_len:]

                # Remove trailing pad tokens
                while (
                    completion_tokens
                    and completion_tokens[-1] == self.tokenizer.pad_token_id
                ):
                    completion_tokens.pop()

                # Check if truncated
                truncated = len(completion_tokens) >= max_tokens and (
                    not completion_tokens
                    or completion_tokens[-1] != self.tokenizer.eos_token_id
                )

                # Extract log probabilities from scores
                token_logprobs = []
                for step_idx, token_id in enumerate(completion_tokens):
                    if step_idx < len(scores):
                        step_logits = scores[step_idx][output_idx]
                        step_log_probs = F.log_softmax(step_logits, dim=-1)
                        token_logprobs.append(step_log_probs[token_id].item())

                # Decode completion text
                completion_text = self.tokenizer.decode(
                    completion_tokens, skip_special_tokens=True
                )

                samples.append(
                    GeneratedSample(
                        text=completion_text,
                        token_ids=completion_tokens,
                        token_logprobs=token_logprobs,
                        truncated=truncated,
                    )
                )

            results.append(
                GenerationResult(
                    key=key,
                    samples=samples,
                    prompt_token_ids=prompt_tokens,
                )
            )

        torch.cuda.empty_cache()
        return results

    # =========================================================================
    # Log Probability Computation
    # =========================================================================

    @torch.inference_mode()
    def _score_sequences(
        self,
        requests: Sequence[SequenceScoringRequest],
    ) -> list[LogProbResult]:
        """Score token sequences and compute log probs for specified ranges.

        This is the core batched scoring method used by all scoring APIs.

        Args:
            requests: Sequence scoring requests (tokens + answer_range).

        Returns:
            LogProbResult for each request with full token_ids, per-token logprobs,
            answer_range, and total_log_prob.
        """
        if not requests:
            return []

        # Prepare batched input with padding
        max_len = max(len(req.tokens) for req in requests)
        pad_token_id = getattr(self.model.config, "pad_token_id", 0) or 0

        batch_input_ids = []
        attention_masks = []

        for req in requests:
            tokens = list(req.tokens)
            padding_needed = max_len - len(tokens)
            padded = tokens + [pad_token_id] * padding_needed
            mask = [1] * len(tokens) + [0] * padding_needed
            batch_input_ids.append(padded)
            attention_masks.append(mask)

        input_ids = torch.tensor(batch_input_ids, device=self.device)
        attention_mask = torch.tensor(attention_masks, device=self.device)

        # Single batched forward pass
        outputs = self.model(input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits  # (batch, seq_len, vocab)

        # Compute log probs
        log_probs_tensor = torch.log_softmax(logits, dim=-1)

        # Build results
        results: list[LogProbResult] = []

        for batch_idx, req in enumerate(requests):
            tokens = list(req.tokens)
            seq_len = len(tokens)
            start, end = req.answer_range

            if seq_len == 0:
                results.append(
                    LogProbResult(
                        token_ids=[],
                        logprobs=[],
                        answer_range=(start, end),
                        total_log_prob=0.0,
                    )
                )
                continue

            # Build per-token logprobs (first token has no context, so None)
            all_logprobs: list[float | None] = [None]
            for pos in range(1, seq_len):
                token_id = tokens[pos]
                # logits at position pos-1 predict token at position pos
                lp = log_probs_tensor[batch_idx, pos - 1, token_id].item()
                all_logprobs.append(lp)

            # Sum logprobs in answer range
            answer_logprobs = [
                all_logprobs[pos]
                for pos in range(start, end)
                if pos < len(all_logprobs) and all_logprobs[pos] is not None
            ]
            total_log_prob = sum(answer_logprobs) if answer_logprobs else 0.0  # type: ignore[arg-type]

            results.append(
                LogProbResult(
                    token_ids=tokens,
                    logprobs=all_logprobs,
                    answer_range=(start, end),
                    total_log_prob=total_log_prob,
                )
            )

        return results

    # =========================================================================
    # Scoring Methods (use batched_map / batched_expand_map)
    # =========================================================================

    def score_tokens(
        self,
        requests: Iterable[ScoreTokensRequest | None],
    ) -> Iterator[ScoreTokensResult]:
        """Score token sequences, yielding results as batches complete.

        Chunks requests internally based on batch_size config.
        Results are yielded in input order. None values are heartbeats
        for timeout-based flushing in streaming mode.
        """

        def process_batch(batch: list[ScoreTokensRequest]) -> list[ScoreTokensResult]:
            score_requests = [
                SequenceScoringRequest(
                    tokens=list(req.prompt_tokens) + list(req.completion_tokens),
                    answer_range=(
                        len(req.prompt_tokens),
                        len(req.prompt_tokens) + len(req.completion_tokens),
                    ),
                )
                for req in batch
            ]
            results = self._score_sequences(score_requests)
            torch.cuda.empty_cache()

            return [
                ScoreTokensResult(key=req.key, log_prob=result.total_log_prob)
                for req, result in zip(batch, results)
            ]

        yield from batched_map(
            requests, process_batch, self._config.batch_size, self._config.flush_timeout
        )

    def score_text(
        self,
        requests: Iterable[ScoreTextRequest | None],
    ) -> Iterator[ScoreTextResult]:
        """Score text continuations (multiple-choice), yielding as complete.

        Sequences are batched at fixed batch_size for GPU efficiency.
        A single request may span multiple batches if it has many choices.
        Results are yielded in input order. None values are heartbeats
        for timeout-based flushing in streaming mode.
        """
        from place_layers.core.token_utils import find_differing_ranges

        def expand(
            req: ScoreTextRequest,
        ) -> tuple[str, list[SequenceScoringRequest]]:
            if not req.choices:
                # Empty list triggers immediate yield of combine(key, [])
                return (req.key, [])

            # Tokenize all (context + choice) sequences
            token_sequences: list[list[int]] = []
            for choice in req.choices:
                ctx_tokens = self.tokenizer.encode(req.context, add_special_tokens=True)
                choice_tokens = self.tokenizer.encode(choice, add_special_tokens=False)
                token_sequences.append(ctx_tokens + choice_tokens)

            # Find answer token ranges using diff-based approach
            answer_ranges = find_differing_ranges(token_sequences)

            # Return key and SequenceScoringRequest elements
            return (
                req.key,
                [
                    SequenceScoringRequest(tokens=list(seq), answer_range=rng)
                    for seq, rng in zip(token_sequences, answer_ranges)
                ],
            )

        def process_batch(
            elements: list[SequenceScoringRequest],
        ) -> list[LogProbResult]:
            return self._score_sequences(elements)

        def combine(key: str, results: list[LogProbResult]) -> ScoreTextResult:
            return ScoreTextResult(key=key, results=results)

        yield from batched_expand_map(
            requests,
            expand,
            process_batch,
            combine,
            self._config.batch_size,
            self._config.flush_timeout,
        )

    def score_chat(
        self,
        requests: Iterable[ScoreChatRequest | None],
    ) -> Iterator[ScoreChatResult]:
        """Score chat completions (multiple-choice), yielding as complete.

        Sequences are batched at fixed batch_size for GPU efficiency.
        A single request may span multiple batches if it has many choices.
        Results are yielded in input order. None values are heartbeats
        for timeout-based flushing in streaming mode.
        """
        from place_layers.core.token_utils import find_differing_ranges

        def expand(
            req: ScoreChatRequest,
        ) -> tuple[str, list[SequenceScoringRequest]]:
            if not req.completion_messages:
                # Empty list triggers immediate yield of combine(key, [])
                return (req.key, [])

            # Tokenize all (context + completion) sequences
            token_sequences: list[list[int]] = []
            for comp_msg in req.completion_messages:
                full_msgs = list(req.context_messages) + [comp_msg]
                full_text = self.tokenizer.apply_chat_template(
                    full_msgs,  # type: ignore[arg-type]
                    tokenize=False,
                    add_generation_prompt=False,
                )
                tokens = self.tokenizer.encode(
                    full_text, add_special_tokens=False  # type: ignore[arg-type]
                )
                token_sequences.append(tokens)

            # Find answer token ranges using diff-based approach
            answer_ranges = find_differing_ranges(token_sequences)

            # Return key and SequenceScoringRequest elements
            return (
                req.key,
                [
                    SequenceScoringRequest(tokens=list(seq), answer_range=rng)
                    for seq, rng in zip(token_sequences, answer_ranges)
                ],
            )

        def process_batch(
            elements: list[SequenceScoringRequest],
        ) -> list[LogProbResult]:
            return self._score_sequences(elements)

        def combine(key: str, results: list[LogProbResult]) -> ScoreChatResult:
            return ScoreChatResult(key=key, results=results)

        yield from batched_expand_map(
            requests,
            expand,
            process_batch,
            combine,
            self._config.batch_size,
            self._config.flush_timeout,
        )

    # =========================================================================
    # Placement
    # =========================================================================

    def set_placement(self, placement: Sequence[str]) -> None:
        """Set mixer placement for Apriel2 supernet model."""
        placement_list = list(placement)

        if self._current_placement == placement_list:
            return

        if self._apriel_structure is None:
            raise ValueError(
                "Model does not support dynamic placement. "
                "Only Apriel2 supernet models support set_placement()."
            )

        if len(placement_list) != self._apriel_structure.num_layers:
            raise ValueError(
                f"placement length ({len(placement_list)}) doesn't match "
                f"model layers ({self._apriel_structure.num_layers})"
            )

        valid_mixers = set(self._apriel_structure.mixer_names)
        for layer_idx, mixer in enumerate(placement_list):
            if mixer not in valid_mixers:
                raise ValueError(
                    f"Invalid mixer '{mixer}' at layer {layer_idx}. "
                    f"Valid mixers: {valid_mixers}"
                )

        for layer_idx, mixer in enumerate(placement_list):
            self._model.model.decoder.blocks[layer_idx].mixer.main_mixer_name = mixer

        self._current_placement = placement_list


class TransformersBackendConfig(BaseBackendConfig):
    """Configuration for Transformers generation backend."""

    _impl_class: ClassVar[type[SyncBackend]] = TransformersBackend

    backend_type: Literal["transformers"] = "transformers"

    model_path: str = Field(description="Path to model (local or HF hub)")
    tokenizer_path: str | None = Field(
        default=None,
        description="Path to tokenizer (defaults to model_path)",
    )
    torch_dtype: Literal["float16", "bfloat16", "float32"] = Field(
        default="bfloat16",
        description="Model dtype",
    )
    device_map: str = Field(
        default="cuda",
        description="Device map for model loading",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Trust remote code when loading model",
    )
    batch_size: int = Field(
        default=1,
        description="Number of prompts to batch together",
        gt=0,
    )
    flush_timeout: float = Field(
        description="Flush partial batches after this many seconds (required for streaming)",
        gt=0,
    )
    revision: str | None = Field(
        default=None,
        description="HuggingFace revision (branch, tag, or commit hash)",
    )
    attn_implementation: str | None = Field(
        description="Attention implementation (e.g., 'eager', 'flash_attention_2'). None uses model default.",
    )
