"""Unit tests for router fitting helpers."""

from __future__ import annotations

import numpy as np
import pytest

from place_layers.evaluator.traces.types import Trace
from place_layers.router.data import (
    PlacementScoreRecord,
    PromptTraceGroup,
    _append_score_record,
    _compute_problem_ids_hash,
    _load_completed_score_records,
    _sample_problem_indices,
)
from place_layers.router.embeddings import (
    load_prompt_embedding_artifact,
    save_prompt_embedding_artifact,
)
from place_layers.router.model import build_placement_features, placement_feature_dim
from place_layers.surrogate.encoding import FeatureExpansionConfig


class TestPlacementFeatures:
    def test_build_placement_features_matches_feature_dim(self) -> None:
        assignments = [
            ("attention", "attention"),
            ("gdn", "attention"),
            ("attention", "gdn"),
        ]
        feature_config = FeatureExpansionConfig.unary()

        features = build_placement_features(
            assignments=assignments,
            mixer_names=["attention", "gdn"],
            feature_config=feature_config,
        )

        assert features.shape == (
            len(assignments),
            placement_feature_dim(
                assignments=assignments,
                mixer_names=["attention", "gdn"],
                feature_config=feature_config,
            ),
        )
        assert features.dtype == np.float32

    def test_build_placement_features_rejects_unknown_mixer(self) -> None:
        with pytest.raises(KeyError):
            build_placement_features(
                assignments=[("attention", "unknown")],
                mixer_names=["attention", "gdn"],
                feature_config=FeatureExpansionConfig.unary(),
            )

    def test_placement_feature_dim_uses_representative_assignment_shape(self) -> None:
        assignments = [
            ("attention", "attention", "gdn"),
            ("gdn", "attention", "gdn"),
        ]
        feature_config = FeatureExpansionConfig(max_order=2, pairwise_max_range=1)

        dim = placement_feature_dim(
            assignments=assignments,
            mixer_names=["attention", "gdn"],
            feature_config=feature_config,
        )
        features = build_placement_features(
            assignments=assignments,
            mixer_names=["attention", "gdn"],
            feature_config=feature_config,
        )
        assert dim == features.shape[1]


class TestRouterEmbeddingArtifacts:
    def test_save_and_load_mean_pooled_prompt_embeddings(self, tmp_path) -> None:
        problems = [
            PromptTraceGroup(
                problem_id="gsm8k:0",
                task_name="gsm8k",
                prompt_tokens=(1, 2),
                traces=(),
            ),
            PromptTraceGroup(
                problem_id="gsm8k:1",
                task_name="gsm8k",
                prompt_tokens=(3,),
                traces=(),
            ),
        ]
        pooled = np.array(
            [
                [2.0, 4.0],
                [2.0, 4.0],
            ],
            dtype=np.float32,
        )
        artifact_path = tmp_path / "train_prompt_embeddings.npz"

        save_prompt_embedding_artifact(
            path=artifact_path,
            problems=problems,
            prompt_embeddings=pooled,
        )
        loaded_embeddings = load_prompt_embedding_artifact(
            path=artifact_path,
            problems=problems,
        )

        np.testing.assert_allclose(
            loaded_embeddings,
            pooled,
        )


class TestRouterScoreResume:
    def test_load_completed_score_records_from_jsonl(self, tmp_path) -> None:
        problems = [
            PromptTraceGroup(
                problem_id="gsm8k:0",
                task_name="gsm8k",
                prompt_tokens=(1, 2),
                traces=(
                    Trace(
                        prompt_tokens=(1, 2),
                        completion_tokens=(3,),
                        correct=True,
                        problem_id="gsm8k:0",
                        task_name="gsm8k",
                        sample_idx=0,
                        teacher_log_prob=-1.0,
                    ),
                ),
            ),
            PromptTraceGroup(
                problem_id="gsm8k:1",
                task_name="gsm8k",
                prompt_tokens=(4, 5),
                traces=(
                    Trace(
                        prompt_tokens=(4, 5),
                        completion_tokens=(6,),
                        correct=False,
                        problem_id="gsm8k:1",
                        task_name="gsm8k",
                        sample_idx=0,
                        teacher_log_prob=-2.0,
                    ),
                ),
            ),
        ]
        placements = [("attention", "attention"), ("gdn", "attention")]
        problem_ids_hash = _compute_problem_ids_hash(
            [problem.problem_id for problem in problems]
        )
        problem_indices = _sample_problem_indices(
            num_problems=len(problems),
            max_problems_per_placement=1,
            seed=42,
            placement_idx=0,
        )
        jsonl_path = tmp_path / "train_placement_scores.jsonl"
        _append_score_record(
            jsonl_path,
            PlacementScoreRecord(
                placement_idx=0,
                placement=["attention", "attention"],
                problem_ids_hash=problem_ids_hash,
                problem_indices=problem_indices,
                scores=[-0.5],
            ),
        )

        score_matrix, observed_mask, completed = _load_completed_score_records(
            jsonl_path=jsonl_path,
            problems=problems,
            placements=placements,
            max_problems_per_placement=1,
            seed=42,
        )

        assert completed.tolist() == [True, False]
        np.testing.assert_allclose(score_matrix[problem_indices, 0], [-0.5])
        assert observed_mask[problem_indices, 0].tolist() == [True]

    def test_load_completed_score_records_rejects_mismatched_hash(
        self, tmp_path
    ) -> None:
        problems = [
            PromptTraceGroup(
                problem_id="gsm8k:0",
                task_name="gsm8k",
                prompt_tokens=(1,),
                traces=(
                    Trace(
                        prompt_tokens=(1,),
                        completion_tokens=(2,),
                        correct=True,
                        problem_id="gsm8k:0",
                        task_name="gsm8k",
                        sample_idx=0,
                        teacher_log_prob=-1.0,
                    ),
                ),
            ),
        ]
        jsonl_path = tmp_path / "train_placement_scores.jsonl"
        _append_score_record(
            jsonl_path,
            PlacementScoreRecord(
                placement_idx=0,
                placement=["attention"],
                problem_ids_hash="wronghash",
                problem_indices=[0],
                scores=[-0.5],
            ),
        )

        with pytest.raises(ValueError, match="does not match the current prompt set"):
            _load_completed_score_records(
                jsonl_path=jsonl_path,
                problems=problems,
                placements=[("attention",)],
                max_problems_per_placement=None,
                seed=42,
            )
