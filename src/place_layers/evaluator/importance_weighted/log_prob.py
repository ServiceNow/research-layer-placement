"""Utilities for computing log-probabilities over token sequences.

This module provides shared functionality for computing log p(completion | prompt)
used by both the trace pre-computation step and the importance-weighted evaluator.
"""

import logging
import time

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@torch.inference_mode()
def compute_completion_log_prob(
    model: torch.nn.Module,
    prompt_tokens: tuple[int, ...],
    completion_tokens: tuple[int, ...],
    device: torch.device | str | None = None,
) -> float:
    """Compute log probability of completion given prompt.

    Performs a forward pass through the model and computes the sum of
    log p(token | prefix) over all completion tokens.

    Args:
        model: Language model with a forward method returning logits.
        prompt_tokens: Tokenized prompt (context).
        completion_tokens: Tokenized completion to score.
        device: Device to place tensors on. If None, uses model.device.

    Returns:
        Sum of log p(token | prefix) over completion tokens.
    """
    if device is None:
        device = model.device

    # Concatenate prompt and completion
    full_tokens = list(prompt_tokens) + list(completion_tokens)
    input_ids = torch.tensor([full_tokens], device=device)

    # Forward pass
    outputs = model(input_ids, use_cache=False)
    logits = outputs.logits  # [1, seq_len, vocab_size]

    # Extract log-probs for completion tokens
    # logits[t] predicts token[t+1], so we need logits[prompt_len-1:-1]
    # for predicting completion tokens at positions prompt_len to end
    prompt_len = len(prompt_tokens)
    completion_len = len(completion_tokens)

    # Positions for prediction: prompt_len-1 to prompt_len+completion_len-2
    # These logits predict tokens at positions: prompt_len to prompt_len+completion_len-1
    pred_logits = logits[0, prompt_len - 1 : prompt_len + completion_len - 1]

    # Target tokens: completion_tokens
    targets = torch.tensor(completion_tokens, device=device)

    # Compute log softmax and gather
    log_probs = F.log_softmax(pred_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.sum().item()


@torch.inference_mode()
def compute_batch_log_probs(
    model: torch.nn.Module,
    traces: list[tuple[tuple[int, ...], tuple[int, ...]]],
    device: torch.device | str | None = None,
    batch_size: int = 8,
    length_tolerance: int = 128,
) -> list[float]:
    """Compute log probabilities for a batch of traces.

    This is more efficient than calling compute_completion_log_prob repeatedly
    when traces have similar lengths, as it batches the forward pass.

    For traces with varying lengths, we process them in groups by total length
    to enable batching while avoiding excessive padding.

    Args:
        model: Language model with a forward method returning logits.
        traces: List of (prompt_tokens, completion_tokens) tuples.
        device: Device to place tensors on. If None, uses model.device.
        batch_size: Max traces per forward pass (higher = faster but more memory).
        length_tolerance: Max token length difference within a batch.

    Returns:
        List of log probabilities in the same order as input traces.
    """
    if device is None:
        device = model.device

    if not traces:
        return []

    # For simplicity and correctness, we'll use a bucketing strategy:
    # Group traces by total length (with some tolerance) to enable batching
    # while keeping padding reasonable.

    # Calculate total lengths and create indexed list
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

    logger.info(f"Computing log-probs for {total_traces} traces (batch_size={batch_size})")

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
            results[idx] = compute_completion_log_prob(
                model, prompt, completion, device
            )
        else:
            # Batch computation with padding
            batch_results = _compute_batch_log_probs_padded(model, batch, device)
            for (idx, _, _, _), log_prob in zip(batch, batch_results):
                results[idx] = log_prob

        processed += len(batch)

        # Release cached GPU memory after each batch.
        # Critical for unified memory systems (Grace-Blackwell) where GPU cache
        # consumes the shared CPU/GPU memory pool. Without this, the cache grows
        # unbounded and causes OOM even though GPU "allocated" memory is low.
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
    model: torch.nn.Module,
    batch: list[tuple[int, tuple[int, ...], tuple[int, ...], int]],
    device: torch.device | str,
) -> list[float]:
    """Compute log probs for a batch with padding.

    Args:
        model: Language model.
        batch: List of (original_idx, prompt_tokens, completion_tokens, total_len).
        device: Device for computation.

    Returns:
        List of log probabilities for the batch.
    """
    # Find max length for padding
    max_len = max(item[3] for item in batch)

    # Prepare padded inputs and metadata
    batch_input_ids = []
    prompt_lengths = []
    completion_lengths = []

    # Get pad token (use 0 if not available)
    pad_token_id = getattr(model.config, "pad_token_id", 0) or 0

    for _, prompt, completion, total_len in batch:
        full_tokens = list(prompt) + list(completion)
        padding_needed = max_len - total_len
        padded = full_tokens + [pad_token_id] * padding_needed
        batch_input_ids.append(padded)
        prompt_lengths.append(len(prompt))
        completion_lengths.append(len(completion))

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = [
        [1] * (pl + cl) + [0] * (max_len - pl - cl)
        for pl, cl in zip(prompt_lengths, completion_lengths)
    ]

    # Forward pass
    input_ids = torch.tensor(batch_input_ids, device=device)
    attn_mask = torch.tensor(attention_mask, device=device)

    outputs = model(input_ids, attention_mask=attn_mask, use_cache=False)
    logits = outputs.logits  # [batch, seq_len, vocab_size]

    # Compute log probs for each item in batch
    results = []
    for batch_idx, (_, prompt, completion, _) in enumerate(batch):
        prompt_len = len(prompt)
        completion_len = len(completion)

        # Extract logits for this item's completion
        pred_logits = logits[
            batch_idx, prompt_len - 1 : prompt_len + completion_len - 1
        ]

        # Target tokens
        targets = torch.tensor(completion, device=device)

        # Compute log softmax and gather
        log_probs = F.log_softmax(pred_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        results.append(token_log_probs.sum().item())

    return results
