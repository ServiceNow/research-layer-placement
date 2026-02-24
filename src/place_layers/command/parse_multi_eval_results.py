"""Parse multi-evaluator exploration results into per-evaluator files.

.. deprecated::
    New exploration runs store per-evaluator score dicts natively in
    ``exploration_results.json``.  This command is only needed for legacy
    exploration runs that stored a single aggregated float score.

Reads exploration_results.json and trace files from a multi-evaluator exploration run,
extracts per-evaluator scores, and generates individual exploration_results.json files
for each evaluator. This enables running placement_landscape on individual evaluators.

Input structure (example):
    {base_dir}/
        exploration_results.json        # Aggregated scores
        exploration_progress.jsonl      # Per-placement assignments + scores
        evals/
            mmlu/
                {placement_1}/validation.jsonl
                {placement_2}/validation.jsonl
                ...
            gsm8k/
                {placement_1}/validation.jsonl
                ...

Output structure:
    {output_dir}/
        mmlu/
            exploration_results.json    # MMLU-only scores
        gsm8k/
            exploration_results.json    # GSM8K-only scores
        ...

Each per-evaluator exploration_results.json contains:
    - assignments: same as aggregated file
    - scores: scores from that evaluator only
    - allocations, num_layers, mixer_names: same as aggregated
    - rho_history, samples_at_check, final_rho: empty (not recomputed)
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import ClassVar, Literal

from hydra.core.hydra_config import HydraConfig
from pydantic import Field

from place_layers.command.base_command import BaseCommand, CommandError
from place_layers.command.base_command_config import BaseCommandConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================


def decode_assignment(placement_code: str, mixer_names: list[str]) -> list[str]:
    """Decode run-length encoded placement string.

    Examples:
        "a2g3" with mixers ["attention", "gdn"] -> ["attention", "attention", "gdn", "gdn", "gdn"]
        "a1g1k2" with mixers ["attention", "gdn", "kda"] -> ["attention", "gdn", "kda", "kda"]

    Args:
        placement_code: Run-length encoded string (e.g., "a1g1k2g1")
        mixer_names: List of mixer names to build abbreviation map

    Returns:
        Decoded assignment as list of mixer names
    """
    # Build reverse map: abbrev -> mixer name (same logic as encoding)
    abbrev_map: dict[str, str] = {}
    used_abbrevs: set[str] = set()
    for mixer in sorted(set(mixer_names)):
        # Find first unused character
        for char in mixer.lower():
            if char not in used_abbrevs:
                abbrev_map[char] = mixer
                used_abbrevs.add(char)
                break
        else:
            # Fallback: first char + number
            base = mixer[0].lower()
            i = 1
            while f"{base}{i}" in used_abbrevs:
                i += 1
            abbrev_map[f"{base}{i}"] = mixer
            used_abbrevs.add(f"{base}{i}")

    # Parse run-length encoding: "a2g3k1" -> [("a", 2), ("g", 3), ("k", 1)]
    # Pattern matches: letter(s) followed by digit(s)
    pattern = r"([a-z]+\d*?)(\d+)"
    matches = re.findall(pattern, placement_code)

    # Expand to assignment
    assignment = []
    for abbrev, count_str in matches:
        mixer = abbrev_map[abbrev]
        assignment.extend([mixer] * int(count_str))

    return assignment


def compute_evaluator_score(trace_dir: Path) -> float:
    """Compute accuracy score for an evaluator from its trace files.

    For likelihood evaluators (MMLU, etc.): reads validation.jsonl and computes
    accuracy from the 'correct' field in each record.

    For trace_likelihood evaluators: aggregates per-problem results.

    Args:
        trace_dir: Directory containing validation.jsonl

    Returns:
        Accuracy score (fraction correct)

    Raises:
        FileNotFoundError: If validation.jsonl doesn't exist
        ValueError: If validation.jsonl is empty or malformed
    """
    validation_file = trace_dir / "validation.jsonl"
    if not validation_file.exists():
        raise FileNotFoundError(f"Missing validation.jsonl in {trace_dir}")

    # Track per-problem data for trace_likelihood records
    problem_log_probs: dict[str, list[float]] = {}
    problem_correct: dict[str, bool | None] = {}

    # Track single-record metrics
    total = 0
    correct = 0

    with validation_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            # Handle different record types
            record_type = record.get("record_type")

            if record_type in ("chat_likelihood", "text_likelihood"):
                # Multiple-choice likelihood records
                total += 1
                if record.get("correct", False):
                    correct += 1

            elif record_type == "generation":
                # Generative evaluator records
                total += 1
                if record.get("correct", False):
                    correct += 1

            elif record_type == "trace_likelihood_problem":
                # Trace likelihood per-problem record
                # For now, we just use mean log prob as a proxy
                # In the future, might want to convert to accuracy somehow
                problem_id = record.get("problem_id")
                mean_log_prob = record.get("mean_log_prob")
                if problem_id and mean_log_prob is not None:
                    if problem_id not in problem_log_probs:
                        problem_log_probs[problem_id] = []
                    problem_log_probs[problem_id].append(mean_log_prob)

            elif record_type in ("trace", "skipped"):
                # Skip trace precomputation records and skipped problems
                continue

            else:
                # Unknown record type - skip silently (avoid 280k warnings)
                pass

    # If we have single-record metrics, use those
    if total > 0:
        return correct / total

    # If we have trace_likelihood records, compute mean log prob
    # Note: This is NOT accuracy, but placement_landscape can still use it
    if problem_log_probs:
        all_log_probs = [lp for lps in problem_log_probs.values() for lp in lps]
        return sum(all_log_probs) / len(all_log_probs) if all_log_probs else 0.0

    raise ValueError(f"No valid records found in {validation_file}")


def process_evaluator(
    evaluator_name: str,
    evals_dir: Path,
    assignments: list[list[str]],
    mixer_names: list[str],
    num_workers: int,
) -> tuple[str, list[list[str]], list[float]]:
    """Process one evaluator: extract scores for all placements.

    Args:
        evaluator_name: Name of the evaluator (e.g., "mmlu")
        evals_dir: Base evals directory containing evaluator subdirs
        assignments: List of assignments (one per placement)
        mixer_names: List of mixer names for decoding
        num_workers: Number of parallel workers for score computation

    Returns:
        Tuple of (evaluator_name, filtered_assignments, scores_list).
        Assignments and scores are filtered together so they stay aligned.
    """
    evaluator_dir = evals_dir / evaluator_name
    if not evaluator_dir.exists():
        logger.warning(f"Evaluator directory not found: {evaluator_dir}")
        return evaluator_name, [], []

    logger.info(f"Processing evaluator: {evaluator_name}")

    # Find all placement directories
    placement_dirs = [d for d in evaluator_dir.iterdir() if d.is_dir()]
    placement_map: dict[tuple[str, ...], Path] = {}

    for placement_dir in placement_dirs:
        placement_code = placement_dir.name
        try:
            decoded = decode_assignment(placement_code, mixer_names)
            placement_map[tuple(decoded)] = placement_dir
        except Exception as e:
            logger.warning(
                f"Failed to decode placement '{placement_code}': {e}"
            )

    # Extract scores for each assignment in order
    scores = []

    # Prepare work items
    work_items = []
    for assignment in assignments:
        assignment_tuple = tuple(assignment)
        if assignment_tuple in placement_map:
            work_items.append((assignment_tuple, placement_map[assignment_tuple]))
        else:
            work_items.append((assignment_tuple, None))

    # Process in parallel with progress tracking
    filtered_assignments: list[list[str]] = []
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {}
            for idx, (assignment_tuple, trace_dir) in enumerate(work_items):
                if trace_dir is not None:
                    future = executor.submit(compute_evaluator_score, trace_dir)
                    future_to_idx[future] = idx

            # Collect results in order
            results: list[float | None] = [None] * len(work_items)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    score = future.result()
                    results[idx] = score
                except Exception as e:
                    logger.error(
                        f"Failed to compute score for placement {idx}: {e}"
                    )
                    results[idx] = None

            # Filter assignments and results together to maintain alignment
            for assignment, result in zip(assignments, results):
                if result is not None:
                    filtered_assignments.append(assignment)
                    scores.append(result)
    else:
        # Sequential processing
        for assignment_tuple, trace_dir in work_items:
            if trace_dir is not None:
                try:
                    score = compute_evaluator_score(trace_dir)
                    scores.append(score)
                    filtered_assignments.append(list(assignment_tuple))
                except Exception as e:
                    logger.error(
                        f"Failed to compute score for {trace_dir}: {e}"
                    )
            else:
                logger.warning(
                    f"Missing trace directory for assignment: {assignment_tuple}"
                )

    logger.info(f"Extracted {len(scores)} scores for {evaluator_name}")
    return evaluator_name, filtered_assignments, scores


# ============================================================================
# Command
# ============================================================================


class ParseMultiEvalResultsCommand(BaseCommand):
    """Parse multi-evaluator exploration results into per-evaluator files."""

    config: "ParseMultiEvalResultsCommandConfig"

    async def run(self) -> None:
        """Execute parsing and generate per-evaluator result files."""
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)
        logger.info("Output directory: %s", output_dir)

        # Load aggregated exploration results
        input_path = Path(self.config.exploration_dir)
        results_file = input_path / "exploration_results.json"

        if not results_file.exists():
            raise CommandError(
                f"exploration_results.json not found in {input_path}"
            )

        logger.info(f"Loading exploration results from {results_file}")
        with results_file.open() as f:
            aggregated_results = json.load(f)

        # Extract common fields
        assignments = aggregated_results["assignments"]
        allocations = aggregated_results["allocations"]
        num_layers = aggregated_results["num_layers"]
        mixer_names = aggregated_results["mixer_names"]
        num_samples = aggregated_results["num_samples"]
        termination_reason = aggregated_results["termination_reason"]
        feature_config = aggregated_results.get("feature_config")

        logger.info(
            f"Loaded {num_samples} assignments with {num_layers} layers, "
            f"mixers: {mixer_names}"
        )

        # Check for evals directory
        evals_dir = input_path / "evals"
        if not evals_dir.exists():
            raise CommandError(
                f"evals directory not found in {input_path}. "
                "This command requires a multi-evaluator exploration run."
            )

        # Find all evaluators
        evaluators = [d.name for d in evals_dir.iterdir() if d.is_dir()]
        if not evaluators:
            raise CommandError(f"No evaluator directories found in {evals_dir}")

        logger.info(f"Found {len(evaluators)} evaluators: {evaluators}")

        # Process each evaluator in parallel
        evaluator_results: dict[str, tuple[list[list[str]], list[float]]] = {}

        if self.config.num_workers > 1 and len(evaluators) > 1:
            logger.info(f"Processing evaluators with {self.config.num_workers} workers")
            with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = {
                    executor.submit(
                        process_evaluator,
                        evaluator_name,
                        evals_dir,
                        assignments,
                        mixer_names,
                        self.config.num_score_workers,
                    ): evaluator_name
                    for evaluator_name in evaluators
                }

                for future in as_completed(futures):
                    evaluator_name = futures[future]
                    try:
                        name, filtered_assigns, scores = future.result()
                        evaluator_results[name] = (filtered_assigns, scores)
                    except Exception as e:
                        logger.error(
                            f"Failed to process evaluator {evaluator_name}: {e}"
                        )
        else:
            # Sequential processing
            for evaluator_name in evaluators:
                name, filtered_assigns, scores = process_evaluator(
                    evaluator_name,
                    evals_dir,
                    assignments,
                    mixer_names,
                    self.config.num_score_workers,
                )
                evaluator_results[name] = (filtered_assigns, scores)

        # Write per-evaluator result files
        for evaluator_name, (eval_assignments, scores) in evaluator_results.items():
            if not scores:
                logger.warning(
                    f"No scores extracted for {evaluator_name}, skipping"
                )
                continue

            # Create output subdirectory
            eval_output_dir = output_dir / evaluator_name
            eval_output_dir.mkdir(parents=True, exist_ok=True)

            # Build per-evaluator result with aligned assignments and scores
            per_eval_result = {
                "assignments": eval_assignments,
                "scores": scores,
                "allocations": allocations,
                "num_layers": num_layers,
                "mixer_names": mixer_names,
                "num_samples": len(scores),
                "termination_reason": termination_reason,
                "feature_config": feature_config,
                # Leave rho fields empty - not recomputed (set to 0.0 for compatibility)
                "rho_history": [],
                "samples_at_check": [],
                "final_rho": 0.0,
            }

            # Write result file
            output_file = eval_output_dir / "exploration_results.json"
            with output_file.open("w") as f:
                json.dump(per_eval_result, f, indent=2)

            logger.info(
                f"Wrote {len(scores)} scores for {evaluator_name} to {output_file}"
            )

        logger.info(
            f"Successfully parsed {len(evaluator_results)} evaluators"
        )


# ============================================================================
# Config
# ============================================================================


class ParseMultiEvalResultsCommandConfig(BaseCommandConfig):
    """Configuration for parse_multi_eval_results command."""

    _impl_class: ClassVar[type[BaseCommand]] = ParseMultiEvalResultsCommand

    command: Literal["parse_multi_eval_results"] = "parse_multi_eval_results"

    exploration_dir: str = Field(
        description="Path to exploration directory containing exploration_results.json and evals/"
    )

    num_workers: int = Field(
        default=4,
        description="Number of parallel workers for processing evaluators",
        gt=0,
    )

    num_score_workers: int = Field(
        default=1,
        description="Number of parallel workers for computing scores within each evaluator",
        gt=0,
    )
