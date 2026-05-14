from __future__ import annotations

import numpy as np
import pytest

import place_layers.command.fit_reward_model as fit_reward_model_module
from place_layers.backend.transformers import TransformersBackendConfig
from place_layers.command.base_command import CommandError
from place_layers.command.fit_reward_model import FitRewardModelCommandConfig
from place_layers.router.data import PromptTraceGroup
from place_layers.router.embeddings import (
    load_prompt_embedding_artifact,
    save_prompt_embedding_artifact,
)


def _problems() -> list[PromptTraceGroup]:
    return [
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


def _fit_reward_model_config(
    *,
    reward_model: dict[str, object],
    reward_dataset_dir: str,
    train_embeddings_path: str | None,
    do_cache_avg_embedding: bool = False,
) -> FitRewardModelCommandConfig:
    return FitRewardModelCommandConfig(
        command="fit_reward_model",
        backend=TransformersBackendConfig(
            model_path="/tmp/dummy-model",
            flush_timeout=0.1,
            attn_implementation=None,
        ),
        reward_dataset_dir=reward_dataset_dir,
        train_embeddings_path=train_embeddings_path,
        do_cache_avg_embedding=do_cache_avg_embedding,
        reward_model=reward_model,
        optimizer={"optimizer_type": "adam"},
        loss={"loss_type": "mse_loss"},
    )


def test_prompt_embedding_artifact_round_trip(tmp_path) -> None:
    problems = _problems()
    pooled = np.array(
        [
            [2.0, 4.0],
            [6.0, 8.0],
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

    np.testing.assert_allclose(loaded_embeddings, pooled)


def test_fit_reward_model_uses_cached_mean_pooled_embeddings(tmp_path) -> None:
    problems = _problems()
    pooled = np.array(
        [
            [2.0, 4.0],
            [6.0, 8.0],
        ],
        dtype=np.float32,
    )
    artifact_path = tmp_path / "train_prompt_embeddings.npz"
    save_prompt_embedding_artifact(
        path=artifact_path,
        problems=problems,
        prompt_embeddings=pooled,
    )

    config = _fit_reward_model_config(
        reward_model={"model_type": "mean_pool"},
        reward_dataset_dir=str(tmp_path),
        train_embeddings_path=str(artifact_path),
    )

    command = config.create_command()
    prompt_embeddings, attention_mask = command._load_prompt_embeddings(
        problems=problems,
        split_name="train",
        backend=None,  # type: ignore[arg-type]
        teacher_placement=("attention",),
        cached_embeddings_path=str(artifact_path),
        progress_description="unused",
    )

    assert prompt_embeddings.shape == (2, 1, 2)
    assert attention_mask.shape == (2, 1)
    np.testing.assert_allclose(prompt_embeddings[:, 0, :], pooled)
    np.testing.assert_array_equal(attention_mask, np.ones((2, 1), dtype=bool))


def test_explicit_cached_embedding_path_takes_precedence_over_do_cache_flag(
    tmp_path,
    monkeypatch,
) -> None:
    problems = _problems()
    pooled = np.array(
        [
            [2.0, 4.0],
            [6.0, 8.0],
        ],
        dtype=np.float32,
    )
    artifact_path = tmp_path / "train_prompt_embeddings.npz"
    save_prompt_embedding_artifact(
        path=artifact_path,
        problems=problems,
        prompt_embeddings=pooled,
    )

    def _should_not_run(**kwargs) -> np.ndarray:
        raise AssertionError("extract_prompt_embeddings should not be called")

    monkeypatch.setattr(
        fit_reward_model_module,
        "extract_prompt_embeddings",
        _should_not_run,
    )

    config = _fit_reward_model_config(
        reward_model={"model_type": "mean_pool"},
        reward_dataset_dir=str(tmp_path),
        train_embeddings_path=str(artifact_path),
        do_cache_avg_embedding=True,
    )

    command = config.create_command()
    prompt_embeddings, attention_mask = command._load_prompt_embeddings(
        problems=problems,
        split_name="train",
        backend=None,  # type: ignore[arg-type]
        teacher_placement=("attention",),
        cached_embeddings_path=str(artifact_path),
        progress_description="unused",
    )

    np.testing.assert_allclose(prompt_embeddings[:, 0, :], pooled)
    np.testing.assert_array_equal(attention_mask, np.ones((2, 1), dtype=bool))


def test_fit_reward_model_saves_computed_mean_pooled_embeddings(
    tmp_path,
    monkeypatch,
) -> None:
    problems = _problems()
    pooled = np.array(
        [
            [2.0, 4.0],
            [6.0, 8.0],
        ],
        dtype=np.float32,
    )
    artifact_path = tmp_path / "train_prompt_embeddings.npz"

    def _fake_extract_prompt_embeddings(**kwargs) -> np.ndarray:
        assert kwargs["output_type"] == "mean_pool"
        return pooled

    monkeypatch.setattr(
        fit_reward_model_module,
        "extract_prompt_embeddings",
        _fake_extract_prompt_embeddings,
    )

    config = _fit_reward_model_config(
        reward_model={"model_type": "mean_pool"},
        reward_dataset_dir=str(tmp_path),
        train_embeddings_path=str(artifact_path),
    )

    command = config.create_command()
    prompt_embeddings, attention_mask = command._load_prompt_embeddings(
        problems=problems,
        split_name="train",
        backend=None,  # type: ignore[arg-type]
        teacher_placement=("attention",),
        cached_embeddings_path=str(artifact_path),
        progress_description="unused",
    )

    assert artifact_path.exists()
    loaded_embeddings = load_prompt_embedding_artifact(
        path=artifact_path,
        problems=problems,
    )
    np.testing.assert_allclose(loaded_embeddings, pooled)
    np.testing.assert_allclose(prompt_embeddings[:, 0, :], pooled)
    np.testing.assert_array_equal(attention_mask, np.ones((2, 1), dtype=bool))


def test_do_cache_avg_embedding_uses_standard_hydra_run_path(tmp_path) -> None:
    config = _fit_reward_model_config(
        reward_model={"model_type": "mean_pool"},
        reward_dataset_dir=str(tmp_path),
        train_embeddings_path=None,
        do_cache_avg_embedding=True,
    )

    command = config.create_command()
    resolved_path = command._resolve_cached_embeddings_path(
        split_name="train",
        output_dir=tmp_path,
    )

    assert resolved_path == str(tmp_path / "train_prompt_embeddings.npz")


def test_fit_reward_model_rejects_cached_embeddings_for_attention_pool(
    tmp_path,
) -> None:
    problems = _problems()
    pooled = np.array(
        [
            [2.0, 4.0],
            [6.0, 8.0],
        ],
        dtype=np.float32,
    )
    artifact_path = tmp_path / "train_prompt_embeddings.npz"
    save_prompt_embedding_artifact(
        path=artifact_path,
        problems=problems,
        prompt_embeddings=pooled,
    )

    config = _fit_reward_model_config(
        reward_model={"model_type": "attention_pool"},
        reward_dataset_dir=str(tmp_path),
        train_embeddings_path=str(artifact_path),
    )

    command = config.create_command()
    with pytest.raises(CommandError, match="mean_pool"):
        command._load_prompt_embeddings(
            problems=problems,
            split_name="train",
            backend=None,  # type: ignore[arg-type]
            teacher_placement=("attention",),
            cached_embeddings_path=str(artifact_path),
            progress_description="unused",
        )
