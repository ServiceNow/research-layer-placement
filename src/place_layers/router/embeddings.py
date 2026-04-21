"""Prompt embedding extraction helpers for reward modeling."""

from __future__ import annotations

import logging
from pathlib import Path
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch

from place_layers.backend.transformers import TransformersBackend
from place_layers.evaluator.progress import create_progress
from place_layers.router.data import PromptTraceGroup

logger = logging.getLogger(__name__)


def save_prompt_embedding_artifact(
    *,
    path: Path,
    problems: Sequence[PromptTraceGroup],
    prompt_embeddings: np.ndarray,
) -> None:
    """Save mean-pooled prompt embeddings for later reuse."""
    embeddings = np.asarray(prompt_embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("prompt_embeddings must be a 2D array")
    if embeddings.shape[0] != len(problems):
        raise ValueError(
            "prompt_embeddings row count must match the number of problems"
        )

    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("wb") as f:
        np.savez(
            f,
            problem_ids=np.asarray(
                [problem.problem_id for problem in problems],
                dtype=np.str_,
            ),
            prompt_embeddings=embeddings,
        )
    tmp_path.replace(path)


def load_prompt_embedding_artifact(
    *,
    path: Path,
    problems: Sequence[PromptTraceGroup],
) -> np.ndarray:
    """Load mean-pooled prompt embeddings and validate prompt alignment."""
    with np.load(path, allow_pickle=False) as data:
        problem_ids = data["problem_ids"].tolist()
        prompt_embeddings = np.asarray(
            data["prompt_embeddings"],
            dtype=np.float32,
        )

    expected_problem_ids = [problem.problem_id for problem in problems]
    if problem_ids != expected_problem_ids:
        raise ValueError(
            f"Prompt embedding artifact {path} does not match the current problem ids"
        )
    if prompt_embeddings.ndim != 2:
        raise ValueError(
            f"Expected prompt_embeddings to be 2D in prompt embedding artifact {path}"
        )
    if prompt_embeddings.shape[0] != len(problems):
        raise ValueError(
            f"Prompt embedding row count does not match problems in artifact {path}"
        )
    return prompt_embeddings


@torch.no_grad()
def extract_prompt_embeddings(
    *,
    problems: Sequence[PromptTraceGroup],
    backend: TransformersBackend,
    teacher_placement: Sequence[str],
    batch_size: int,
    output_type: Literal["token", "mean_pool"] = "token",
    progress_description: str = "Extracting prompt embeddings",
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Extract prompt embeddings, either token-level or mean-pooled."""
    if output_type == "mean_pool":
        if not problems:
            return np.empty((0, 0), dtype=np.float32)
    else:
        if not problems:
            return (
                np.empty((0, 0, 0), dtype=np.float32),
                np.empty((0, 0), dtype=bool),
            )

    model = backend.model
    backend.set_placement(teacher_placement)
    device = backend.device
    max_prompt_len = max(len(problem.prompt_tokens) for problem in problems)

    logger.info(
        "Extracting %s prompt embeddings for %d prompts",
        "mean-pooled" if output_type == "mean_pool" else "token-level",
        len(problems),
    )
    pooled_embeddings: np.ndarray | None = None
    token_embeddings: np.ndarray | None = None
    attention_mask_full = np.zeros((len(problems), max_prompt_len), dtype=bool)

    with create_progress() as progress:
        task_id = progress.add_task(
            progress_description,
            total=len(problems),
        )
        for start in range(0, len(problems), batch_size):
            batch = problems[start : start + batch_size]
            lengths = [len(problem.prompt_tokens) for problem in batch]
            batch_max_len = max(lengths)

            input_ids = torch.zeros(
                (len(batch), batch_max_len), dtype=torch.long, device=device
            )
            attention_mask = torch.zeros_like(input_ids)

            for row_idx, problem in enumerate(batch):
                tokens = torch.tensor(
                    problem.prompt_tokens, dtype=torch.long, device=device
                )
                length = len(problem.prompt_tokens)
                input_ids[row_idx, :length] = tokens
                attention_mask[row_idx, :length] = 1

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden states")

            end = start + len(batch)
            if output_type == "mean_pool":
                last_hidden = hidden_states[-1].detach().to(torch.float32)
                mask = attention_mask.unsqueeze(-1).to(torch.float32)
                batch_pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                batch_pooled_np = batch_pooled.cpu().numpy()
                if pooled_embeddings is None:
                    pooled_embeddings = np.zeros(
                        (len(problems), int(batch_pooled_np.shape[-1])),
                        dtype=np.float32,
                    )
                pooled_embeddings[start:end, :] = batch_pooled_np
            else:
                last_hidden = hidden_states[-1].detach().cpu().to(torch.float32).numpy()
                if token_embeddings is None:
                    hidden_dim = int(last_hidden.shape[-1])
                    token_embeddings = np.zeros(
                        (len(problems), max_prompt_len, hidden_dim),
                        dtype=np.float32,
                    )
                token_embeddings[start:end, :batch_max_len, :] = last_hidden
                attention_mask_full[start:end, :batch_max_len] = (
                    attention_mask.detach().cpu().numpy().astype(bool)
                )
            progress.advance(task_id, len(batch))

    if output_type == "mean_pool":
        if pooled_embeddings is None:
            raise RuntimeError("Prompt embedding extraction produced no embeddings")
        return pooled_embeddings
    if token_embeddings is None:
        raise RuntimeError("Prompt embedding extraction produced no embeddings")
    return token_embeddings, attention_mask_full
