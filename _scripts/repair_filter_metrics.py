#!/usr/bin/env python3
"""Repair validation.jsonl files where truncated samples have correct_by_filter=null.

Bug: truncated samples were written with correct_by_filter=null instead of
{filter_name: False, ...}, causing summarize_generations_by_filter to exclude
them from per-filter accuracy computation (inflating filter metrics).

This script:
1. Finds all validation.jsonl files under the outputs directory
2. For each file, infers filter names from records that have them
3. Backfills records with correct_by_filter=null using all-False
4. Rewrites validation.jsonl in-place
5. Re-generates the sibling evaluation_results.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path so we can import project code
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def repair_validation_jsonl(validation_path: Path, *, dry_run: bool = False) -> bool:
    """Repair a single validation.jsonl file.

    Returns True if any records were modified.
    """
    lines = validation_path.read_text().splitlines()
    records = []
    for line in lines:
        line = line.strip()
        if line:
            records.append(json.loads(line))

    if not records:
        return False

    # Only look at generation/trace records (those with correct_by_filter field)
    gen_records = [r for r in records if "correct_by_filter" in r]
    if not gen_records:
        return False

    # Collect filter names from records that have them
    filter_names: set[str] = set()
    for r in gen_records:
        cbf = r.get("correct_by_filter")
        if cbf and isinstance(cbf, dict):
            filter_names.update(cbf.keys())

    if not filter_names:
        # No records have filter data at all — nothing to backfill from
        return False

    # Find records that need repair: correct_by_filter is null
    needs_repair = [r for r in gen_records if r.get("correct_by_filter") is None]
    if not needs_repair:
        return False

    sorted_filter_names = sorted(filter_names)
    primary = sorted_filter_names[0]

    print(f"  {validation_path.parent.name}: {len(needs_repair)}/{len(gen_records)} records need repair")
    print(f"    Filter names: {sorted_filter_names}")

    if dry_run:
        return True

    # Patch the records in place
    for r in needs_repair:
        r["correct_by_filter"] = {k: False for k in sorted_filter_names}
        r["filtered_completion_text_by_filter"] = {k: "" for k in sorted_filter_names}
        if r.get("primary_filter") is None:
            r["primary_filter"] = primary

    # Rewrite the file
    with validation_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    return True


def regenerate_evaluation_results(eval_dir: Path) -> None:
    """Re-generate evaluation_results.json from the repaired validation.jsonl.

    eval_dir is the directory containing evaluation_results.json (parent of the
    trace subdirectory).
    """
    eval_results_path = eval_dir / "evaluation_results.json"
    if not eval_results_path.exists():
        return

    # Load existing results to preserve non-filter fields
    existing = json.loads(eval_results_path.read_text())

    # Find the trace subdirectory (contains validation.jsonl)
    trace_dirs = [d for d in eval_dir.iterdir() if d.is_dir() and (d / "validation.jsonl").exists()]
    if not trace_dirs:
        return

    from place_layers.evaluator.traces.writer import summarize_generations_by_filter

    trace_dir = trace_dirs[0]
    try:
        summaries = summarize_generations_by_filter(trace_dir)
    except Exception as e:
        print(f"    Warning: failed to re-summarize {trace_dir}: {e}")
        return

    # Update filter metrics
    existing["filter_problem_accuracy"] = {
        name: s.problem_accuracy for name, s in summaries.items()
    }
    existing["filter_sample_accuracy"] = {
        name: s.sample_accuracy for name, s in summaries.items()
    }
    existing["extra"] = {
        "filter_task_stats": {
            name: {
                task_name: {
                    "problem_accuracy": stats.problem_accuracy,
                    "sample_accuracy": stats.sample_accuracy,
                    "num_problems": stats.num_problems,
                    "num_samples": stats.num_samples,
                }
                for task_name, stats in s.task_stats.items()
            }
            for name, s in summaries.items()
        }
    }

    eval_results_path.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"    Updated {eval_results_path.name}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "outputs_dir",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent.parent / "outputs",
        help="Root outputs directory to scan (default: ./outputs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be fixed without modifying files",
    )
    args = parser.parse_args()

    outputs_dir: Path = args.outputs_dir.resolve()
    if not outputs_dir.exists():
        print(f"Error: {outputs_dir} does not exist")
        sys.exit(1)

    validation_files = sorted(outputs_dir.rglob("validation.jsonl"))
    print(f"Found {len(validation_files)} validation.jsonl files under {outputs_dir}")

    repaired = 0
    for vf in validation_files:
        if repair_validation_jsonl(vf, dry_run=args.dry_run):
            repaired += 1
            if not args.dry_run:
                # eval_dir is two levels up: outputs/.../eval_dir/trace_subdir/validation.jsonl
                eval_dir = vf.parent.parent
                regenerate_evaluation_results(eval_dir)

    print(f"\n{'Would repair' if args.dry_run else 'Repaired'} {repaired}/{len(validation_files)} files")


if __name__ == "__main__":
    main()
