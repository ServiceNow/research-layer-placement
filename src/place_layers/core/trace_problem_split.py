"""Shared helpers for splitting a combined trace dataset by problem."""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel

from place_layers.evaluator.evaluators import EvaluatorsConfig
from place_layers.evaluator.objective import IndependentObjectiveConfig
from place_layers.evaluator.trace_likelihood import TraceLikelihoodEvaluatorConfig
from place_layers.evaluator.traces.types import Trace, TraceDataset


logger = logging.getLogger(__name__)


class PerProblemTraceSpec(BaseModel):
    """Mapping from synthesized objective name to source problem."""

    objective_name: str
    problem_id: str
    task_name: str
    num_traces: int
    trace_dir: str


class PerProblemTraceManifest(BaseModel):
    """Manifest describing the per-problem evaluator split."""

    items: list[PerProblemTraceSpec]


def _sanitize_fragment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return cleaned or "problem"


def _problem_sort_key(problem_id: str) -> tuple[str, int | str]:
    task_name, _, suffix = problem_id.partition(":")
    try:
        return task_name, int(suffix)
    except ValueError:
        return task_name, suffix


def _group_traces_by_problem(trace_sources: list[str]) -> dict[str, list[Trace]]:
    grouped: dict[str, list[Trace]] = defaultdict(list)

    for source in trace_sources:
        dataset = TraceDataset.load_from_directory(Path(source))
        for trace in dataset.traces:
            grouped[trace.problem_id].append(trace)

    if not grouped:
        raise ValueError("No traces found in trace_sources for per-problem command")

    return dict(grouped)


def build_per_problem_evaluators(
    base_evaluators: EvaluatorsConfig,
    split_root: Path,
) -> tuple[EvaluatorsConfig, PerProblemTraceManifest]:
    """Split one trace-likelihood evaluator into one evaluator per problem."""
    if len(base_evaluators.configs) != 1:
        raise ValueError(
            "Per-problem commands require exactly one base evaluator config"
        )

    base_name, base_cfg = next(iter(base_evaluators.configs.items()))
    if not isinstance(base_cfg, TraceLikelihoodEvaluatorConfig):
        raise ValueError(
            "Per-problem commands currently require a trace_likelihood evaluator"
        )

    grouped = _group_traces_by_problem(base_cfg.trace_sources)

    if split_root.exists():
        shutil.rmtree(split_root)
    split_root.mkdir(parents=True, exist_ok=True)

    configs = {}
    items: list[PerProblemTraceSpec] = []
    for index, problem_id in enumerate(sorted(grouped, key=_problem_sort_key)):
        traces = grouped[problem_id]
        objective_name = f"problem_{index:03d}"
        problem_dir = split_root / f"{objective_name}_{_sanitize_fragment(problem_id)}"
        problem_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "teacher_model": "per_problem_split",
            "samples_per_problem": len(traces),
        }
        (problem_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        with (problem_dir / "validation.jsonl").open("w") as handle:
            for trace in traces:
                handle.write(trace.model_dump_json())
                handle.write("\n")

        configs[objective_name] = base_cfg.model_copy(
            update={"trace_sources": [str(problem_dir)]}
        )
        items.append(
            PerProblemTraceSpec(
                objective_name=objective_name,
                problem_id=problem_id,
                task_name=traces[0].task_name,
                num_traces=len(traces),
                trace_dir=str(problem_dir),
            )
        )

    logger.info(
        "Prepared %d per-problem trace evaluators from base evaluator '%s'",
        len(items),
        base_name,
    )
    manifest = PerProblemTraceManifest(items=items)
    return (
        EvaluatorsConfig(
            configs=configs,
            objective=IndependentObjectiveConfig(),
        ),
        manifest,
    )
