"""TransformersBackend - Sync backend using HuggingFace Transformers."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, ClassVar, Literal, Self, Sequence

import torch
import torch.nn.functional as F
from pydantic import Field

from place_layers.backend.base import BaseBackendConfig, SyncBackend
from place_layers.backend.types import ChatMessage, LogProbResult, ModelStructure
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


class TransformersBackend:
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
        self._current_placement: tuple[str, ...] | None = None

    @property
    def batch_size(self) -> int:
        """Number of prompts to batch together."""
        return self._config.batch_size

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
            "device_map": self.config.device_map,
            "trust_remote_code": self.config.trust_remote_code,
            "revision": self.config.revision,
        }

        # Add attn_implementation if specified
        if self.config.attn_implementation is not None:
            load_kwargs["attn_implementation"] = self.config.attn_implementation
            logger.info(f"Using attention implementation: {self.config.attn_implementation}")

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

        Raises:
            NotImplementedError: If model is not an Apriel2 supernet.
        """
        if self._apriel_structure is None:
            raise NotImplementedError(
                "Model structure discovery requires an Apriel2 supernet model. "
                "The loaded model does not support dynamic mixer placement."
            )
        return ModelStructure(
            num_layers=self._apriel_structure.num_layers,
            mixer_names=self._apriel_structure.mixer_names,
        )

    # =========================================================================
    # Generation
    # =========================================================================

    def batch_generate(
        self,
        messages_batch: Sequence[list[ChatMessage]],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[GenerationResult]:
        """Generate n completions for each message sequence (truly batched).

        Tokenizes all message sequences together, runs a single batched generate call,
        and extracts log probabilities from generation scores.
        """
        if not messages_batch:
            return []

        batch_size = len(messages_batch)
        mem_before = get_memory_stats()
        logger.debug(
            f"batch_generate: batch_size={batch_size}, n={n}, max_tokens={max_tokens}, "
            f"GPU alloc={mem_before.gpu_allocated_gb:.2f}GB"
        )

        # Apply chat template to all message sequences
        prompt_texts = []
        for messages in messages_batch:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,  # type: ignore[arg-type]
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_texts.append(prompt_text)

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

        for prompt_idx in range(batch_size):
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
                    request_id="",
                    samples=samples,
                    prompt_token_ids=prompt_tokens,
                )
            )

        torch.cuda.empty_cache()
        return results

    def batch_generate_text(
        self,
        prompts: Sequence[str],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[GenerationResult]:
        """Generate n completions for each text prompt without chat template.

        Similar to generate_batch but skips chat template application.

        Args:
            prompts: List of raw text prompt strings.
            n: Number of completions per prompt.
            max_tokens: Maximum tokens per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.

        Returns:
            List of GenerationResult, one per prompt.
        """
        if not prompts:
            return []

        batch_size = len(prompts)
        mem_before = get_memory_stats()
        logger.debug(
            f"batch_generate_text: batch_size={batch_size}, n={n}, max_tokens={max_tokens}, "
            f"GPU alloc={mem_before.gpu_allocated_gb:.2f}GB"
        )

        # Tokenize all prompts with left padding (required for batched generation)
        # Save original padding side and restore after
        original_padding_side = self._tokenizer.padding_side
        self._tokenizer.padding_side = "left"

        batch_encoding = self._tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
        )

        self._tokenizer.padding_side = original_padding_side

        # Track original prompt lengths (before padding)
        prompt_lengths = [
            (batch_encoding.attention_mask[i] == 1).sum().item()
            for i in range(batch_size)
        ]

        # Move to device
        input_ids = batch_encoding.input_ids.to(self._device)
        attention_mask = batch_encoding.attention_mask.to(self._device)

        # Generate with scores for log probability extraction
        # temperature=0 means greedy (matches vLLM behavior)
        do_sample = temperature > 0
        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature if do_sample else 1.0,  # Avoid temperature=0 warning
                top_p=top_p,
                do_sample=do_sample,
                num_return_sequences=n,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # outputs.sequences: [batch_size * n, seq_len]
        # outputs.scores: tuple of [batch_size * n, vocab_size] for each generated token
        sequences = outputs.sequences
        scores = outputs.scores  # Tuple of logits for each generation step

        # Build results for each prompt
        results: list[GenerationResult] = []
        padded_prompt_len = input_ids.shape[1]

        for prompt_idx in range(batch_size):
            # Get unpadded prompt tokens for this prompt
            prompt_start = padded_prompt_len - prompt_lengths[prompt_idx]
            prompt_tokens = input_ids[prompt_idx, prompt_start:].tolist()

            samples: list[GeneratedSample] = []

            for sample_idx in range(n):
                # Output index in the flattened batch
                output_idx = prompt_idx * n + sample_idx

                # Extract completion tokens (after padded prompt)
                full_sequence = sequences[output_idx].tolist()
                completion_tokens = full_sequence[padded_prompt_len:]

                # Remove trailing pad tokens
                while completion_tokens and completion_tokens[-1] == self._tokenizer.pad_token_id:
                    completion_tokens.pop()

                # Check if truncated
                truncated = (
                    len(completion_tokens) >= max_tokens
                    and (
                        not completion_tokens
                        or completion_tokens[-1] != self._tokenizer.eos_token_id
                    )
                )

                # Extract log probabilities from scores
                token_logprobs = []
                for step_idx, token_id in enumerate(completion_tokens):
                    if step_idx < len(scores):
                        step_logits = scores[step_idx][output_idx]
                        step_log_probs = F.log_softmax(step_logits, dim=-1)
                        token_logprobs.append(step_log_probs[token_id].item())

                # Decode completion text
                completion_text = self._tokenizer.decode(
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
                    request_id="",
                    samples=samples,
                    prompt_token_ids=prompt_tokens,
                )
            )

        # Clear cache after generation
        torch.cuda.empty_cache()

        return results

    def generate_text(
        self,
        prompt: str,
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        request_id: str = "",
    ) -> GenerationResult:
        """Generate n completions from a text prompt without chat template.

        Args:
            prompt: Raw text prompt string.
            n: Number of completions to generate.
            max_tokens: Maximum tokens per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            request_id: Optional ID for tracking.

        Returns:
            GenerationResult with samples and prompt token IDs.
        """
        results = self.batch_generate_text(
            [prompt],
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        result = results[0]
        result.request_id = request_id
        return result

    # =========================================================================
    # Log Probability Computation
    # =========================================================================

    @torch.inference_mode()
    def _score_sequences(
        self,
        sequences: Sequence[Sequence[int]],
        answer_ranges: Sequence[tuple[int, int]],
    ) -> list[LogProbResult]:
        """Score token sequences and compute log probs for specified ranges.

        This is the core batched scoring method. All compute_*_log_probs methods
        call this with appropriate sequences and ranges.

        Args:
            sequences: Token sequences to score.
            answer_ranges: (start, end) ranges for each sequence indicating which
                tokens contribute to total_log_prob. Range is [start, end).

        Returns:
            LogProbResult for each sequence with full token_ids, per-token logprobs,
            answer_range, and total_log_prob.
        """
        if not sequences:
            return []

        # Prepare batched input with padding
        max_len = max(len(seq) for seq in sequences)
        pad_token_id = getattr(self.model.config, "pad_token_id", 0) or 0

        batch_input_ids = []
        attention_masks = []

        for seq in sequences:
            seq_list = list(seq)
            padding_needed = max_len - len(seq_list)
            padded = seq_list + [pad_token_id] * padding_needed
            mask = [1] * len(seq_list) + [0] * padding_needed
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

        for batch_idx, (seq, (start, end)) in enumerate(zip(sequences, answer_ranges)):
            seq_list = list(seq)
            seq_len = len(seq_list)

            if seq_len == 0:
                results.append(
                    LogProbResult(
                        token_ids=(),
                        logprobs=(),
                        answer_range=(start, end),
                        total_log_prob=0.0,
                    )
                )
                continue

            # Build per-token logprobs (first token has no context, so None)
            all_logprobs: list[float | None] = [None]
            for pos in range(1, seq_len):
                token_id = seq_list[pos]
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
                    token_ids=tuple(seq_list),
                    logprobs=tuple(all_logprobs),
                    answer_range=(start, end),
                    total_log_prob=total_log_prob,
                )
            )

        return results

    def compute_token_log_probs(
        self,
        traces: Sequence[tuple[Sequence[int], Sequence[int]]],
    ) -> list[float]:
        """Compute log probabilities for token-based traces.

        Groups traces by length for efficient batched computation.
        """
        if not traces:
            return []

        # Build sequences and ranges from (prompt, completion) pairs
        indexed_items: list[tuple[int, list[int], tuple[int, int]]] = []
        for i, (prompt, completion) in enumerate(traces):
            full_seq = list(prompt) + list(completion)
            answer_range = (len(prompt), len(full_seq))
            indexed_items.append((i, full_seq, answer_range))

        # Sort by length for better batching
        indexed_items.sort(key=lambda x: len(x[1]))

        # Process in batches of similar length
        results = [0.0] * len(traces)
        batch_size = self._config.batch_size
        length_tolerance = 128

        i = 0
        while i < len(indexed_items):
            batch_start = i
            base_length = len(indexed_items[i][1])

            # Extend batch while lengths are within tolerance
            while (
                i < len(indexed_items)
                and i - batch_start < batch_size
                and len(indexed_items[i][1]) - base_length <= length_tolerance
            ):
                i += 1

            batch = indexed_items[batch_start:i]
            batch_seqs = [item[1] for item in batch]
            batch_ranges = [item[2] for item in batch]
            batch_indices = [item[0] for item in batch]

            # Score batch
            log_prob_results = self._score_sequences(batch_seqs, batch_ranges)

            # Extract total_log_prob and restore order
            for idx, result in zip(batch_indices, log_prob_results):
                results[idx] = result.total_log_prob

            torch.cuda.empty_cache()

        return results

    def compute_text_log_probs(
        self,
        text_groups: Sequence[tuple[str, Sequence[str]]],
    ) -> list[LogProbResult]:
        """Compute log probabilities for text continuations with diff-based scoring."""
        if not text_groups:
            return []

        from place_layers.core.token_utils import find_differing_ranges

        all_results: list[LogProbResult] = []

        for context, choices in text_groups:
            if not choices:
                continue

            # Tokenize all (context + choice) sequences for this group
            token_sequences: list[list[int]] = []
            for choice in choices:
                ctx_tokens = self.tokenizer.encode(context, add_special_tokens=True)
                choice_tokens = self.tokenizer.encode(choice, add_special_tokens=False)
                token_sequences.append(ctx_tokens + choice_tokens)

            # Find answer token ranges using diff-based approach
            answer_ranges = find_differing_ranges(token_sequences)

            # Score sequences (batched)
            results = self._score_sequences(token_sequences, answer_ranges)
            all_results.extend(results)

        return all_results

    def compute_chat_log_probs(
        self,
        chat_groups: Sequence[tuple[Sequence[ChatMessage], Sequence[ChatMessage]]],
    ) -> list[LogProbResult]:
        """Compute log probabilities for chat completions with diff-based scoring.

        Args:
            chat_groups: Sequence of (context_messages, completion_messages) tuples.
                context_messages: The prompt (system, user, etc.)
                completion_messages: Multiple assistant messages, one per choice.
        """
        if not chat_groups:
            return []

        from place_layers.core.token_utils import find_differing_ranges

        all_results: list[LogProbResult] = []

        for context_msgs, completion_msgs in chat_groups:
            if not completion_msgs:
                continue

            # Tokenize all (context + choice) sequences for this group
            token_sequences: list[list[int]] = []
            for comp_msg in completion_msgs:
                full_msgs = list(context_msgs) + [comp_msg]
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

            # Score sequences (batched)
            results = self._score_sequences(token_sequences, answer_ranges)
            all_results.extend(results)

        return all_results

    # =========================================================================
    # Placement
    # =========================================================================

    def set_placement(self, mixer_names: Sequence[str]) -> None:
        """Set mixer placement for Apriel2 supernet model."""
        placement_tuple = tuple(mixer_names)

        if self._current_placement == placement_tuple:
            return

        if self._apriel_structure is None:
            raise ValueError(
                "Model does not support dynamic placement. "
                "Only Apriel2 supernet models support set_placement()."
            )

        if len(mixer_names) != self._apriel_structure.num_layers:
            raise ValueError(
                f"mixer_names length ({len(mixer_names)}) doesn't match "
                f"model layers ({self._apriel_structure.num_layers})"
            )

        valid_mixers = set(self._apriel_structure.mixer_names)
        for layer_idx, mixer_name in enumerate(mixer_names):
            if mixer_name not in valid_mixers:
                raise ValueError(
                    f"Invalid mixer '{mixer_name}' at layer {layer_idx}. "
                    f"Valid mixers: {valid_mixers}"
                )

        for layer_idx, mixer_name in enumerate(mixer_names):
            self.model.model.decoder.blocks[layer_idx].mixer.main_mixer_name = (
                mixer_name
            )

        self._current_placement = placement_tuple


class TransformersBackendConfig(BaseBackendConfig):
    """Configuration for Transformers generation backend."""

    _backend_class: ClassVar[type[SyncBackend]] = TransformersBackend  # type: ignore[assignment]

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
    revision: str | None = Field(
        default=None,
        description="HuggingFace revision (branch, tag, or commit hash)",
    )
