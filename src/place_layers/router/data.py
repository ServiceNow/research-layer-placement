"""Data loading and scoring helpers for router fitting."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from place_layers.backend import BackendError, ScoreTokensRequest
from place_layers.backend.base import AsyncBackend, SyncBackend
from place_layers.evaluator.progress import create_progress
from place_layers.evaluator.traces.types import Trace, TraceDataset

logger = logging.getLogger(__name__)


class PromptTraceGroup(BaseModel):
    """All teacher traces associated with one prompt/problem."""

    problem_id: str
    task_name: str
    prompt_tokens: tuple[int, ...]
    traces: tuple[Trace, ...] = Field(default_factory=tuple)

    @property
    def num_traces(self) -> int:
        return len(self.traces)


class PlacementScoreRecord(BaseModel):
    """One completed placement-scoring column for resumable router fitting."""

    model_config = {"frozen": True}

    record_type: Literal["placement_score"] = "placement_score"
    placement_idx: int = Field(description="Column index in the sampled placement list.")
    placement: list[str] = Field(description="Mixer assignment for this placement.")
    problem_ids_hash: str = Field(
        description="Hash of problem ids to validate resume compatibility."
    )
    problem_indices: list[int] = Field(
        description="Prompt indices evaluated for this placement."
    )
    scores: list[float] = Field(
        description="Mean log-probability for the sampled prompts under this placement."
    )


@dataclass(frozen=True)
class RewardDatasetSplit:
    """One train/test split of a reward dataset."""

    problems: list[PromptTraceGroup]
    score_matrix: np.ndarray
    observed_mask: np.ndarray

    @property
    def problem_ids(self) -> list[str]:
        return [problem.problem_id for problem in self.problems]


def save_sampled_placements(
    *,
    path: Path,
    placements: Sequence[Sequence[str]],
) -> None:
    """Save sampled placements in the legacy fit_router JSON format."""
    serialized = [list(placement) for placement in placements]
    path.write_text(json.dumps(serialized, indent=2))


def load_sampled_placements(path: Path) -> list[tuple[str, ...]]:
    """Load sampled placements from the legacy fit_router JSON format."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of placements in {path}")

    placements: list[tuple[str, ...]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, list) or not all(isinstance(m, str) for m in item):
            raise ValueError(f"Invalid placement at index {idx} in {path}")
        placements.append(tuple(item))
    return placements


def load_prompt_trace_groups(output_dir: Path) -> list[PromptTraceGroup]:
    """Load precomputed teacher traces and group them by problem."""
    dataset = TraceDataset.load_from_directory(output_dir)
    grouped: dict[str, list[Trace]] = defaultdict(list)
    for trace in dataset.traces:
        grouped[trace.problem_id].append(trace)

    problems: list[PromptTraceGroup] = []
    for problem_id in sorted(grouped):
        traces = grouped[problem_id]
        first = traces[0]
        prompt_tokens = first.prompt_tokens
        task_name = first.task_name
        for trace in traces[1:]:
            if trace.prompt_tokens != prompt_tokens:
                raise ValueError(
                    f"Problem {problem_id} has inconsistent prompt tokens across traces"
                )
            if trace.task_name != task_name:
                raise ValueError(
                    f"Problem {problem_id} has inconsistent task names across traces"
                )
        problems.append(
            PromptTraceGroup(
                problem_id=problem_id,
                task_name=task_name,
                prompt_tokens=prompt_tokens,
                traces=tuple(traces),
            )
        )

    logger.info(
        "Loaded %d prompts from %s",
        len(problems),
        output_dir,
    )
    return problems


def load_reward_dataset_split_from_score_log(
    *,
    jsonl_path: Path,
    problems: Sequence[PromptTraceGroup],
    placements: Sequence[Sequence[str]],
    max_problems_per_placement: int | None,
    seed: int,
) -> RewardDatasetSplit:
    """Load a reward dataset split from legacy score logs plus placements."""
    score_matrix, observed_mask, _ = _load_completed_score_records(
        jsonl_path=jsonl_path,
        problems=problems,
        placements=placements,
        max_problems_per_placement=max_problems_per_placement,
        seed=seed,
    )
    return RewardDatasetSplit(
        problems=list(problems),
        score_matrix=score_matrix,
        observed_mask=observed_mask,
    )


def sample_random_placements(
    *,
    mixer_names: Sequence[str],
    num_layers: int,
    num_placements: int,
    seed: int,
    include_teacher: bool = True,
    teacher_mixer_name: str = "attention",
) -> list[tuple[str, ...]]:
    """Sample unique random placements uniformly over layers."""
    rng = random.Random(seed)
    placements: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()

    if include_teacher:
        if teacher_mixer_name not in mixer_names:
            raise ValueError(
                f"Teacher mixer '{teacher_mixer_name}' is not in mixer_names={list(mixer_names)}"
            )
        teacher = tuple([teacher_mixer_name] * num_layers)
        placements.append(teacher)
        seen.add(teacher)

    max_attempts = max(100, num_placements * 50)
    attempts = 0
    while len(placements) < num_placements and attempts < max_attempts:
        placement = tuple(rng.choice(mixer_names) for _ in range(num_layers))
        attempts += 1
        if placement in seen:
            continue
        placements.append(placement)
        seen.add(placement)

    if len(placements) < num_placements:
        raise ValueError(
            f"Could only sample {len(placements)} unique placements after {attempts} attempts"
        )

    return placements


async def _iter_score_tokens(
    backend: AsyncBackend | SyncBackend,
    requests: Iterable[ScoreTokensRequest],
):
    """Yield score results from either backend type."""
    if isinstance(backend, AsyncBackend):
        async for result in backend.score_tokens(requests):
            yield result
        return

    for result in backend.score_tokens(requests):
        yield result


async def _set_backend_placement(
    backend: AsyncBackend | SyncBackend,
    placement: Sequence[str],
) -> None:
    """Set placement on either backend type."""
    if isinstance(backend, AsyncBackend):
        await backend.set_placement(placement)
        return
    backend.set_placement(placement)


async def _score_problem_traces_once(
    backend: AsyncBackend | SyncBackend,
    problems: Sequence[PromptTraceGroup],
    on_problem_complete: Callable[[], None] | None = None,
) -> np.ndarray:
    """Score all traces once at the backend's current placement."""
    problem_to_idx = {
        problem.problem_id: problem_idx for problem_idx, problem in enumerate(problems)
    }
    expected_counts = np.array(
        [len(problem.traces) for problem in problems],
        dtype=np.int32,
    )
    seen_counts = np.zeros(len(problems), dtype=np.int32)
    log_prob_sums = np.zeros(len(problems), dtype=np.float64)

    def make_requests() -> Iterable[ScoreTokensRequest]:
        for problem in problems:
            for trace in problem.traces:
                yield ScoreTokensRequest(
                    key=problem.problem_id,
                    prompt_tokens=list(trace.prompt_tokens),
                    completion_tokens=list(trace.completion_tokens),
                )

    async for result in _iter_score_tokens(backend, make_requests()):
        problem_id = result.key
        if isinstance(result, BackendError):
            raise RuntimeError(
                f"Backend failed while scoring {problem_id}: "
                f"{result.error_type} - {result.message}"
            )

        problem_idx = problem_to_idx[problem_id]
        seen_counts[problem_idx] += 1
        if seen_counts[problem_idx] > expected_counts[problem_idx]:
            raise RuntimeError(f"Received too many scoring results for {problem_id}")
        log_prob_sums[problem_idx] += float(result.log_prob)
        if (
            on_problem_complete is not None
            and seen_counts[problem_idx] == expected_counts[problem_idx]
        ):
            on_problem_complete()

    if not np.array_equal(seen_counts, expected_counts):
        missing = [
            problems[idx].problem_id
            for idx in range(len(problems))
            if seen_counts[idx] != expected_counts[idx]
        ]
        raise RuntimeError(
            "Incomplete scoring results for problems: "
            + ", ".join(missing[:5])
            + ("..." if len(missing) > 5 else "")
        )

    return (log_prob_sums / expected_counts).astype(np.float32)


def _compute_problem_ids_hash(problem_ids: Sequence[str]) -> str:
    payload = json.dumps(list(problem_ids), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _sample_problem_indices(
    *,
    num_problems: int,
    max_problems_per_placement: int | None,
    seed: int,
    placement_idx: int,
) -> list[int]:
    if num_problems <= 0:
        return []
    if (
        max_problems_per_placement is None
        or max_problems_per_placement >= num_problems
    ):
        return list(range(num_problems))

    sample_size = max(1, max_problems_per_placement)
    rng = random.Random(seed + placement_idx)
    return sorted(rng.sample(range(num_problems), sample_size))


def _load_completed_score_records(
    *,
    jsonl_path: Path,
    problems: Sequence[PromptTraceGroup],
    placements: Sequence[Sequence[str]],
    max_problems_per_placement: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score_matrix = np.zeros((len(problems), len(placements)), dtype=np.float32)
    observed_mask = np.zeros((len(problems), len(placements)), dtype=bool)
    completed = np.zeros(len(placements), dtype=bool)
    if not jsonl_path.exists():
        return score_matrix, observed_mask, completed

    expected_hash = _compute_problem_ids_hash(
        [problem.problem_id for problem in problems]
    )
    with jsonl_path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = PlacementScoreRecord.model_validate_json(line)
            if record.problem_ids_hash != expected_hash:
                raise ValueError(
                    f"Score log {jsonl_path} line {line_num} does not match the current prompt set"
                )
            if not (0 <= record.placement_idx < len(placements)):
                raise ValueError(
                    f"Score log {jsonl_path} line {line_num} has invalid placement index {record.placement_idx}"
                )
            expected_placement = list(placements[record.placement_idx])
            if record.placement != expected_placement:
                raise ValueError(
                    f"Score log {jsonl_path} line {line_num} placement does not match sampled placements"
                )
            expected_problem_indices = _sample_problem_indices(
                num_problems=len(problems),
                max_problems_per_placement=max_problems_per_placement,
                seed=seed,
                placement_idx=record.placement_idx,
            )
            if record.problem_indices != expected_problem_indices:
                raise ValueError(
                    f"Score log {jsonl_path} line {line_num} prompt subsample does not match current settings"
                )

            column = np.asarray(record.scores, dtype=np.float32)
            if column.shape != (len(record.problem_indices),):
                raise ValueError(
                    f"Score log {jsonl_path} line {line_num} has invalid score length {column.shape}"
                )

            if completed[record.placement_idx]:
                existing = score_matrix[record.problem_indices, record.placement_idx]
                if not np.allclose(existing, column):
                    raise ValueError(
                        f"Score log {jsonl_path} line {line_num} duplicates placement {record.placement_idx} with different values"
                    )
                continue

            score_matrix[record.problem_indices, record.placement_idx] = column
            observed_mask[record.problem_indices, record.placement_idx] = True
            completed[record.placement_idx] = True

    return score_matrix, observed_mask, completed


def _append_score_record(path: Path, record: PlacementScoreRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(record.model_dump_json() + "\n")


async def score_problem_trace_matrix(
    *,
    backend: AsyncBackend | SyncBackend,
    problems: Sequence[PromptTraceGroup],
    placements: Sequence[Sequence[str]],
    progress_description: str = "Scoring placements",
    progress_jsonl_path: Path | None = None,
    max_problems_per_placement: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Score all prompt/placement pairs as mean log-prob over teacher traces.

    When ``progress_jsonl_path`` is provided, callers may rely on the resumable
    JSONL side effect and observed mask even if they do not use the returned scores.
    """
    if progress_jsonl_path is not None:
        score_matrix, observed_mask, completed = _load_completed_score_records(
            jsonl_path=progress_jsonl_path,
            problems=problems,
            placements=placements,
            max_problems_per_placement=max_problems_per_placement,
            seed=seed,
        )
        if completed.any():
            logger.info(
                "Resuming %s from %d/%d completed placements",
                progress_description,
                int(completed.sum()),
                len(placements),
            )
    else:
        score_matrix = np.zeros((len(problems), len(placements)), dtype=np.float32)
        observed_mask = np.zeros((len(problems), len(placements)), dtype=bool)
        completed = np.zeros(len(placements), dtype=bool)

    problem_ids_hash = _compute_problem_ids_hash(
        [problem.problem_id for problem in problems]
    )
    with create_progress() as progress:
        placements_task_id = progress.add_task(
            progress_description,
            total=len(placements),
            completed=int(completed.sum()),
        )
        for placement_idx, is_completed in enumerate(completed):
            if not is_completed:
                continue
            problem_indices = _sample_problem_indices(
                num_problems=len(problems),
                max_problems_per_placement=max_problems_per_placement,
                seed=seed,
                placement_idx=placement_idx,
            )
            progress.add_task(
                f"{progress_description} placement {placement_idx + 1}/{len(placements)}",
                total=len(problem_indices),
                completed=len(problem_indices),
            )
        for placement_idx, placement in enumerate(placements):
            if completed[placement_idx]:
                continue
            problem_indices = _sample_problem_indices(
                num_problems=len(problems),
                max_problems_per_placement=max_problems_per_placement,
                seed=seed,
                placement_idx=placement_idx,
            )
            sampled_problems = [problems[idx] for idx in problem_indices]
            logger.info(
                "Scoring placement %d/%d on %d/%d prompts",
                placement_idx + 1,
                len(placements),
                len(problem_indices),
                len(problems),
            )
            placement_task_id = progress.add_task(
                f"{progress_description} placement {placement_idx + 1}/{len(placements)}",
                total=len(problem_indices),
                completed=0,
            )
            await _set_backend_placement(backend, placement)
            score_matrix[problem_indices, placement_idx] = await _score_problem_traces_once(
                backend,
                sampled_problems,
                on_problem_complete=lambda: progress.advance(placement_task_id),
            )
            observed_mask[problem_indices, placement_idx] = True
            if progress_jsonl_path is not None:
                _append_score_record(
                    progress_jsonl_path,
                    PlacementScoreRecord(
                        placement_idx=placement_idx,
                        placement=list(placement),
                        problem_ids_hash=problem_ids_hash,
                        problem_indices=problem_indices,
                        scores=score_matrix[problem_indices, placement_idx]
                        .astype(float)
                        .tolist(),
                    ),
                )
            completed[placement_idx] = True
            progress.advance(placements_task_id)
    return score_matrix, observed_mask
