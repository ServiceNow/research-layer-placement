from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from place_layers.evaluator.lm_eval_utils import (
    apply_all_filters,
    check_correctness_all_filters,
)


class _ReplaceFilter:
    def __init__(self, old: str, new: str) -> None:
        self._old = old
        self._new = new

    def apply(self, resps: list[list[str]], docs: list[Any]) -> list[list[str]]:
        return [[r.replace(self._old, self._new) for r in doc_resps] for doc_resps in resps]


class _SuffixFilter:
    def __init__(self, suffix: str) -> None:
        self._suffix = suffix

    def apply(self, resps: list[list[str]], docs: list[Any]) -> list[list[str]]:
        return [[r + self._suffix for r in doc_resps] for doc_resps in resps]


@dataclass(frozen=True)
class _FilterEnsemble:
    name: str
    filters: list[Callable[[], Any]]


class _DummyTask:
    def __init__(self, filters: list[_FilterEnsemble]) -> None:
        self._filters = filters

    def process_results(self, doc: dict, resps: list[str]) -> dict[str, int]:
        # Mimic lm-eval "exact_match" metric.
        return {"exact_match": 1 if resps and resps[0] == doc["gold"] else 0}


def test_apply_all_filters_none_defined() -> None:
    task = _DummyTask(filters=[])
    completion = "raw"
    assert apply_all_filters(task, completion) == {"none": "raw"}


def test_check_correctness_all_filters_multiple() -> None:
    task = _DummyTask(
        filters=[
            _FilterEnsemble(
                name="strict-match",
                filters=[
                    lambda: _ReplaceFilter("Answer: ", ""),
                    lambda: _ReplaceFilter("#### ", ""),
                ],
            ),
            _FilterEnsemble(
                name="wrong",
                filters=[lambda: _SuffixFilter("!!!")],
            ),
        ]
    )

    doc = {"gold": "42"}
    completion = "Answer: #### 42"

    results = check_correctness_all_filters(task, doc, completion)

    assert set(results.keys()) == {"strict-match", "wrong"}
    assert results["strict-match"] == (True, "42")
    assert results["wrong"] == (False, "Answer: #### 42!!!")

