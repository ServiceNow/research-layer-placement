"""Low-level Transformers client for model operations.

This module provides a context manager for HuggingFace model lifecycle
and low-level operations like generation and log probability computation.
Analogous to vllm_client.py for the vLLM HTTP API.

Usage:
    config = TransformersClientConfig(model_path="/path/to/model")
    with TransformersClient(config) as client:
        messages_batch = [[{"role": "user", "content": "Hello"}], ...]
        results = client.generate_batch(messages_batch, n=8, max_tokens=512, ...)
        log_probs = client.compute_batch_log_probs(traces, batch_size=8)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Self, Sequence

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

from place_layers.backend.vllm import (
    GeneratedSample,
    GenerationResult,
)
from place_layers.core.memory_logging import (
    log_memory,
    log_memory_delta,
    get_memory_stats,
    MemoryMonitor,
    force_garbage_collection,
    log_gpu_memory_summary,
    log_tensor_info,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class TransformersClientConfig(BaseModel):
    """Configuration for TransformersClient."""

    model_config = {"frozen": True, "extra": "forbid"}

    model_path: str = Field(description="Path to model (local or HF hub)")
    tokenizer_path: str | None = Field(
        default=None,
        description="Path to tokenizer (defaults to model_path)",
    )
    torch_dtype: str = Field(
        default="bfloat16",
        description="Model dtype (float16, bfloat16, float32)",
    )
    device_map: str = Field(
        default="cuda",
        description="Device map for model loading",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Trust remote code when loading model",
    )
    revision: str | None = Field(
        default=None,
        description="HuggingFace revision (branch, tag, or commit hash)",
    )


class TransformersClient:
    """Context manager for HuggingFace model operations.

    Handles model and tokenizer lifecycle (load on enter, unload on exit),
    and provides methods for generation and log probability computation.
    """

    def __init__(self, config: TransformersClientConfig) -> None:
        self.config = config
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._device: torch.device | None = None

    @property
    def model(self) -> "PreTrainedModel":
        """Get the loaded model (raises if not in context)."""
        if self._model is None:
            raise RuntimeError("Client not initialized. Use as context manager.")
        return self._model

    @property
    def tokenizer(self) -> "PreTrainedTokenizerBase":
        """Get the loaded tokenizer (raises if not in context)."""
        if self._tokenizer is None:
            raise RuntimeError("Client not initialized. Use as context manager.")
        return self._tokenizer

    @property
    def device(self) -> torch.device:
        """Get the model's device (raises if not in context)."""
        if self._device is None:
            raise RuntimeError("Client not initialized. Use as context manager.")
        return self._device

    def __enter__(self) -> Self:
        """Load model and tokenizer."""
        from transformers import AutoTokenizer

        logger.info(f"Loading model from {self.config.model_path}...")
        log_memory("MODEL LOAD START")
        mem_before = get_memory_stats()
        start_time = time.perf_counter()

        # Load model using flexible loader
        self._model = self._load_model()
        self._model.eval()
        self._device = next(self._model.parameters()).device

        # Load tokenizer
        tokenizer_path = self.config.tokenizer_path or self.config.model_path
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=self.config.trust_remote_code,
            revision=self.config.revision,
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
        """Load the model, detecting appropriate class from config.

        Supports both CausalLM and ImageTextToText models by inspecting
        the model's auto_map configuration.
        """
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
        )

        # Load config to determine model class from auto_map
        config = AutoConfig.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code,
            revision=self.config.revision,
        )
        auto_map = getattr(config, "auto_map", {})
        architectures = getattr(config, "architectures", [])

        if "AutoModelForImageTextToText" in auto_map or "LlavaForConditionalGeneration" in architectures:
            model_class = AutoModelForImageTextToText
        elif "AutoModelForCausalLM" in auto_map:
            model_class = AutoModelForCausalLM
        else:
            # Default to CausalLM for standard models without auto_map
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
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)

        model = model_class.from_pretrained(
            self.config.model_path,
            torch_dtype=torch_dtype,
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code,
            revision=self.config.revision,
        )
        return model

    # =========================================================================
    # Generation
    # =========================================================================

    def generate_batch(
        self,
        messages_batch: Sequence[list[dict[str, str]]],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[GenerationResult]:
        """Generate n completions for each message sequence (truly batched).

        Tokenizes all message sequences together, runs a single batched generate call,
        and extracts log probabilities from generation scores (no extra forward passes).

        Args:
            messages_batch: List of message sequences (each is list of role/content dicts).
            n: Number of completions per sequence.
            max_tokens: Maximum tokens per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.

        Returns:
            List of GenerationResult, one per message sequence.
        """
        if not messages_batch:
            return []

        batch_size = len(messages_batch)
        mem_before = get_memory_stats()
        logger.debug(
            f"generate_batch: batch_size={batch_size}, n={n}, max_tokens={max_tokens}, "
            f"GPU alloc={mem_before.gpu_allocated_gb:.2f}GB"
        )

        # Apply chat template to all message sequences
        prompt_texts = []
        for messages in messages_batch:
            prompt_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(prompt_text)

        # Tokenize all prompts with left padding (required for batched generation)
        # Save original padding side and restore after
        original_padding_side = self._tokenizer.padding_side
        self._tokenizer.padding_side = "left"

        batch_encoding = self._tokenizer(
            prompt_texts,
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

    # =========================================================================
    # Log Probability Computation
    # =========================================================================

    @torch.inference_mode()
    def compute_log_prob(
        self,
        prompt_tokens: tuple[int, ...] | list[int],
        completion_tokens: tuple[int, ...] | list[int],
    ) -> float:
        """Compute log probability of completion given prompt.

        Performs a forward pass through the model and computes the sum of
        log p(token | prefix) over all completion tokens.

        Args:
            prompt_tokens: Tokenized prompt (context).
            completion_tokens: Tokenized completion to score.

        Returns:
            Sum of log p(token | prefix) over completion tokens.
        """
        # Concatenate prompt and completion
        full_tokens = list(prompt_tokens) + list(completion_tokens)
        input_ids = torch.tensor([full_tokens], device=self._device)

        # Forward pass
        outputs = self._model(input_ids, use_cache=False)
        logits = outputs.logits  # [1, seq_len, vocab_size]

        # Extract log-probs for completion tokens
        prompt_len = len(prompt_tokens)
        completion_len = len(completion_tokens)

        # logits[t] predicts token[t+1]
        pred_logits = logits[0, prompt_len - 1 : prompt_len + completion_len - 1]

        # Target tokens
        targets = torch.tensor(list(completion_tokens), device=self._device)

        # Compute log softmax and gather
        log_probs = F.log_softmax(pred_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        return token_log_probs.sum().item()

    @torch.inference_mode()
    def compute_batch_log_probs(
        self,
        traces: list[tuple[tuple[int, ...], tuple[int, ...]]],
        batch_size: int = 8,
        length_tolerance: int = 128,
    ) -> list[float]:
        """Compute log probabilities for a batch of traces.

        More efficient than calling compute_log_prob repeatedly when traces
        have similar lengths, as it batches the forward pass.

        Args:
            traces: List of (prompt_tokens, completion_tokens) tuples.
            batch_size: Max traces per forward pass.
            length_tolerance: Max token length difference within a batch.

        Returns:
            List of log probabilities in the same order as input traces.
        """
        if not traces:
            return []

        # Create indexed list with total lengths
        indexed_traces = [
            (i, prompt, completion, len(prompt) + len(completion))
            for i, (prompt, completion) in enumerate(traces)
        ]

        # Sort by length for better batching
        indexed_traces.sort(key=lambda x: x[3])

        # Process in batches of similar length
        results = [0.0] * len(traces)
        total_traces = len(indexed_traces)
        processed = 0
        num_batches = 0
        start_time = time.perf_counter()
        last_log_time = start_time

        logger.info(
            f"Computing log-probs for {total_traces} traces (batch_size={batch_size})"
        )

        i = 0
        while i < len(indexed_traces):
            # Find batch boundaries
            batch_start = i
            base_length = indexed_traces[i][3]

            # Extend batch while lengths are within tolerance
            while (
                i < len(indexed_traces)
                and i - batch_start < batch_size
                and indexed_traces[i][3] - base_length <= length_tolerance
            ):
                i += 1

            batch = indexed_traces[batch_start:i]
            num_batches += 1

            if len(batch) == 1:
                # Single item, use simple computation
                idx, prompt, completion, _ = batch[0]
                results[idx] = self.compute_log_prob(prompt, completion)
            else:
                # Batch computation with padding
                batch_results = self._compute_batch_log_probs_padded(batch)
                for (idx, _, _, _), log_prob in zip(batch, batch_results):
                    results[idx] = log_prob

            processed += len(batch)

            # Release cached GPU memory after each batch
            torch.cuda.empty_cache()

            # Log progress every 10 seconds
            now = time.perf_counter()
            if now - last_log_time >= 10.0:
                elapsed = now - start_time
                rate = processed / elapsed
                eta = (total_traces - processed) / rate if rate > 0 else 0
                logger.info(
                    f"Progress: {processed}/{total_traces} traces "
                    f"({100*processed/total_traces:.1f}%), "
                    f"{rate:.1f} traces/sec, ETA {eta:.0f}s"
                )
                last_log_time = now

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Completed {total_traces} traces in {num_batches} batches, "
            f"{elapsed:.1f}s ({total_traces/elapsed:.1f} traces/sec)"
        )

        return results

    def _compute_batch_log_probs_padded(
        self,
        batch: list[tuple[int, tuple[int, ...], tuple[int, ...], int]],
    ) -> list[float]:
        """Compute log probs for a batch with padding."""
        # Find max length for padding
        max_len = max(item[3] for item in batch)

        # Prepare padded inputs
        batch_input_ids = []
        prompt_lengths = []
        completion_lengths = []

        pad_token_id = getattr(self._model.config, "pad_token_id", 0) or 0

        for _, prompt, completion, total_len in batch:
            full_tokens = list(prompt) + list(completion)
            padding_needed = max_len - total_len
            padded = full_tokens + [pad_token_id] * padding_needed
            batch_input_ids.append(padded)
            prompt_lengths.append(len(prompt))
            completion_lengths.append(len(completion))

        # Create attention mask
        attention_mask = [
            [1] * (pl + cl) + [0] * (max_len - pl - cl)
            for pl, cl in zip(prompt_lengths, completion_lengths)
        ]

        # Forward pass
        input_ids = torch.tensor(batch_input_ids, device=self._device)
        attn_mask = torch.tensor(attention_mask, device=self._device)

        outputs = self._model(input_ids, attention_mask=attn_mask, use_cache=False)
        logits = outputs.logits

        # Compute log probs for each item
        results = []
        for batch_idx, (_, prompt, completion, _) in enumerate(batch):
            prompt_len = len(prompt)
            completion_len = len(completion)

            pred_logits = logits[
                batch_idx, prompt_len - 1 : prompt_len + completion_len - 1
            ]
            targets = torch.tensor(list(completion), device=self._device)

            log_probs = F.log_softmax(pred_logits, dim=-1)
            token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

            results.append(token_log_probs.sum().item())

        return results
