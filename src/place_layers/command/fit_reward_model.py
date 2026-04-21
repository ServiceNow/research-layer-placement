"""Fit a prompt-conditioned reward model from a generated reward dataset."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import torch
import yaml
from hydra.core.hydra_config import HydraConfig
from pydantic import BaseModel, Field

from place_layers.backend.transformers import (
    TransformersBackend,
    TransformersBackendConfig,
)
from place_layers.backend.types import SupernetStructure
from place_layers.command.base_command import BaseCommand, CommandError
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.router import (
    RewardModelConfig,
    RewardModelFitResult,
    RewardOptimizerConfig,
    build_checkpoint_payload,
    build_placement_features,
    evaluate_reward_model,
    fit_reward_model,
    load_reward_dataset_split_from_score_log,
    load_prompt_trace_groups,
    load_sampled_placements,
    PromptTraceGroup,
    placement_feature_dim,
    extract_prompt_embeddings,
    load_prompt_embedding_artifact,
    save_prompt_embedding_artifact,
)
from place_layers.surrogate.encoding import FeatureExpansionConfig

logger = logging.getLogger(__name__)


class LegacyRewardDatasetConfig(BaseModel):
    """Dataset settings recovered from the generator Hydra config."""

    train_trace_dir: str
    test_trace_dir: str
    seed: int
    max_problems_per_placement: int | None = None
    teacher_mixer_name: str


def _resolve_training_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise CommandError("training_device=cuda was requested but CUDA is not available")
    return torch.device(device_name)


class FitRewardModelCommand(BaseCommand):
    """Fit a configurable reward model using extracted or cached prompt embeddings."""

    config: "FitRewardModelCommandConfig"

    async def run(self) -> None:
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_dir = Path(self.config.reward_dataset_dir)
        placements_path = dataset_dir / "sampled_placements.json"
        train_scores_path = dataset_dir / "train_placement_scores.jsonl"
        test_scores_path = dataset_dir / "test_placement_scores.jsonl"

        dataset_config = self._load_dataset_config(dataset_dir)
        placements = load_sampled_placements(placements_path)
        if not placements:
            raise CommandError("Reward dataset has no sampled placements")

        train_problems = load_prompt_trace_groups(Path(dataset_config.train_trace_dir))
        test_problems = load_prompt_trace_groups(Path(dataset_config.test_trace_dir))
        train_split = load_reward_dataset_split_from_score_log(
            jsonl_path=train_scores_path,
            problems=train_problems,
            placements=placements,
            max_problems_per_placement=dataset_config.max_problems_per_placement,
            seed=dataset_config.seed,
        )
        test_split = load_reward_dataset_split_from_score_log(
            jsonl_path=test_scores_path,
            problems=test_problems,
            placements=placements,
            max_problems_per_placement=dataset_config.max_problems_per_placement,
            seed=dataset_config.seed,
        )
        if not train_split.problems:
            raise CommandError("Training reward dataset split is empty")
        if not test_split.problems:
            raise CommandError("Test reward dataset split is empty")

        backend_cm = (
            self.config.backend.create_backend()
            if self.config.backend is not None
            else nullcontext()
        )
        with backend_cm as backend:
            validated_backend = self._validate_backend(
                backend=backend,
                placements=placements,
                teacher_mixer_name=dataset_config.teacher_mixer_name,
            )
            structure = validated_backend.model_structure
            if not isinstance(structure, SupernetStructure):
                raise CommandError("fit_reward_model requires a supernet backend")
            mixer_names = list(structure.mixer_names)
            num_layers = structure.num_layers
            teacher_placement = tuple(
                [dataset_config.teacher_mixer_name] * num_layers
            )
            train_token_embeddings, train_attention_mask = (
                self._load_prompt_embeddings(
                    problems=train_split.problems,
                    split_name="train",
                    backend=validated_backend,
                    teacher_placement=teacher_placement,
                    cached_embeddings_path=self._resolve_cached_embeddings_path(
                        split_name="train",
                        output_dir=output_dir,
                    ),
                    progress_description="Train prompt embeddings",
                )
            )
            test_token_embeddings, test_attention_mask = (
                self._load_prompt_embeddings(
                    problems=test_split.problems,
                    split_name="test",
                    backend=validated_backend,
                    teacher_placement=teacher_placement,
                    cached_embeddings_path=self._resolve_cached_embeddings_path(
                        split_name="test",
                        output_dir=output_dir,
                    ),
                    progress_description="Test prompt embeddings",
                )
            )

        placement_features = build_placement_features(
            assignments=placements,
            mixer_names=mixer_names,
            feature_config=self.config.feature_config,
        )
        train_device = _resolve_training_device(self.config.training_device)
        model, target_mean, target_std, train_loss_history = fit_reward_model(
            prompt_token_embeddings=train_token_embeddings,
            prompt_attention_mask=train_attention_mask,
            placement_features=placement_features,
            score_matrix=train_split.score_matrix,
            observed_mask=train_split.observed_mask,
            reward_model_config=self.config.reward_model,
            optimizer_config=self.config.optimizer,
            num_epochs=self.config.num_epochs,
            batch_size=self.config.batch_size,
            log_every_n_steps=self.config.log_every_n_steps,
            seed=self.config.seed,
            device=train_device,
        )

        train_metrics, _ = evaluate_reward_model(
            model,
            prompt_token_embeddings=train_token_embeddings,
            prompt_attention_mask=train_attention_mask,
            placement_features=placement_features,
            score_matrix=train_split.score_matrix,
            observed_mask=train_split.observed_mask,
            batch_size=self.config.eval_batch_size,
            device=train_device,
            target_mean=target_mean,
            target_std=target_std,
        )
        test_metrics, _ = evaluate_reward_model(
            model,
            prompt_token_embeddings=test_token_embeddings,
            prompt_attention_mask=test_attention_mask,
            placement_features=placement_features,
            score_matrix=test_split.score_matrix,
            observed_mask=test_split.observed_mask,
            batch_size=self.config.eval_batch_size,
            device=train_device,
            target_mean=target_mean,
            target_std=target_std,
        )

        fit_result = RewardModelFitResult(
            model_type=self.config.reward_model.model_type,
            optimizer_type=self.config.optimizer.optimizer_type,
            num_epochs=self.config.num_epochs,
            batch_size=self.config.batch_size,
            num_train_examples=int(train_split.observed_mask.sum()),
            num_test_examples=int(test_split.observed_mask.sum()),
            num_train_prompts=len(train_split.problems),
            num_test_prompts=len(test_split.problems),
            num_placements=len(placements),
            embedding_dim=int(train_token_embeddings.shape[-1]),
            placement_feature_dim=placement_feature_dim(
                assignments=placements,
                mixer_names=mixer_names,
                feature_config=self.config.feature_config,
            ),
            feature_config=self.config.feature_config,
            target_mean=target_mean,
            target_std=target_std,
            train_loss_history=train_loss_history,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
        )

        checkpoint_path = output_dir / "reward_model.pt"
        result_path = output_dir / "reward_model_fit_result.json"
        torch.save(
            build_checkpoint_payload(
                model=model.cpu(),
                reward_model_config=self.config.reward_model,
                optimizer_config=self.config.optimizer,
                mixer_names=mixer_names,
                feature_config=self.config.feature_config,
                embedding_dim=int(train_token_embeddings.shape[-1]),
                placement_feature_dim=int(placement_features.shape[-1]),
                target_mean=target_mean,
                target_std=target_std,
            ),
            checkpoint_path,
        )
        result_path.write_text(fit_result.model_dump_json(indent=2))

        logger.info("Saved reward model checkpoint to %s", checkpoint_path)
        logger.info("Saved reward model fit summary to %s", result_path)
        logger.info(
            "Test metrics: RMSE=%.4f MAE=%.4f Spearman=%.4f mean_prompt_spearman=%.4f",
            fit_result.test_metrics.rmse,
            fit_result.test_metrics.mae,
            fit_result.test_metrics.spearman,
            fit_result.test_metrics.mean_prompt_spearman,
        )

    def _validate_backend(
        self,
        *,
        backend: TransformersBackend | None,
        placements: list[tuple[str, ...]],
        teacher_mixer_name: str,
    ) -> TransformersBackend:
        if backend is None:
            raise CommandError("fit_reward_model requires a transformers backend")
        if not isinstance(backend, TransformersBackend):
            raise CommandError("fit_reward_model only supports backend=transformers")

        structure = backend.model_structure
        if not isinstance(structure, SupernetStructure):
            raise CommandError("fit_reward_model requires a supernet backend")
        if teacher_mixer_name not in structure.mixer_names:
            raise CommandError(
                f"Teacher mixer '{teacher_mixer_name}' is not available in "
                f"{list(structure.mixer_names)}"
            )
        valid_mixers = set(structure.mixer_names)
        for placement_idx, placement in enumerate(placements):
            if len(placement) != structure.num_layers:
                raise CommandError(
                    f"Placement {placement_idx} has {len(placement)} layers, "
                    f"expected {structure.num_layers}"
                )
            invalid_mixers = sorted(set(placement) - valid_mixers)
            if invalid_mixers:
                raise CommandError(
                    f"Placement {placement_idx} contains unknown mixers {invalid_mixers}"
                )
        return backend

    def _load_prompt_embeddings(
        self,
        *,
        problems: list[PromptTraceGroup],
        split_name: str,
        backend: TransformersBackend,
        teacher_placement: tuple[str, ...],
        cached_embeddings_path: str | None,
        progress_description: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.config.reward_model.model_type == "mean_pool":
            if cached_embeddings_path is not None:
                artifact_path = Path(cached_embeddings_path)
                if artifact_path.exists():
                    pooled_embeddings = load_prompt_embedding_artifact(
                        path=artifact_path,
                        problems=problems,
                    )
                    logger.info(
                        "Loaded cached %s mean-pooled prompt embeddings from %s",
                        split_name,
                        cached_embeddings_path,
                    )
                    return (
                        pooled_embeddings[:, None, :],
                        np.ones((pooled_embeddings.shape[0], 1), dtype=bool),
                    )

            pooled_embeddings = extract_prompt_embeddings(
                problems=problems,
                backend=backend,
                teacher_placement=teacher_placement,
                batch_size=self.config.embedding_batch_size,
                output_type="mean_pool",
                progress_description=progress_description,
            )
            if cached_embeddings_path is not None:
                save_prompt_embedding_artifact(
                    path=Path(cached_embeddings_path),
                    problems=problems,
                    prompt_embeddings=pooled_embeddings,
                )
                logger.info(
                    "Saved %s mean-pooled prompt embeddings to %s",
                    split_name,
                    cached_embeddings_path,
                )
            return (
                pooled_embeddings[:, None, :],
                np.ones((pooled_embeddings.shape[0], 1), dtype=bool),
            )

        if cached_embeddings_path is not None:
            raise CommandError(
                "Cached mean-pooled prompt embeddings are only supported for "
                "reward_model.model_type=mean_pool"
            )

        token_embeddings, attention_mask = extract_prompt_embeddings(
            problems=problems,
            backend=backend,
            teacher_placement=teacher_placement,
            batch_size=self.config.embedding_batch_size,
            output_type="token",
            progress_description=progress_description,
        )
        return token_embeddings, attention_mask

    def _resolve_cached_embeddings_path(
        self,
        *,
        split_name: str,
        output_dir: Path,
    ) -> str | None:
        explicit_path = (
            self.config.train_embeddings_path
            if split_name == "train"
            else self.config.test_embeddings_path
        )
        if explicit_path is not None:
            return explicit_path
        if self.config.do_cache_avg_embedding:
            return str(output_dir / f"{split_name}_prompt_embeddings.npz")
        return None

    def _load_dataset_config(self, dataset_dir: Path) -> LegacyRewardDatasetConfig:
        hydra_config_path = dataset_dir / ".hydra" / "config.yaml"
        if not hydra_config_path.exists():
            raise CommandError(
                f"Missing Hydra config at {hydra_config_path}; "
                "fit_reward_model needs the dataset command config to recover trace paths"
            )

        payload = yaml.safe_load(hydra_config_path.read_text())
        if not isinstance(payload, dict):
            raise CommandError(f"Invalid Hydra config at {hydra_config_path}")

        try:
            return LegacyRewardDatasetConfig.model_validate(payload)
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise CommandError(
                f"Could not read reward dataset settings from {hydra_config_path}: {exc}"
            ) from exc


class FitRewardModelCommandConfig(BaseCommandConfig):
    """Configuration for fit_reward_model."""

    _impl_class: ClassVar[type[BaseCommand]] = FitRewardModelCommand

    command: Literal["fit_reward_model"] = "fit_reward_model"

    backend: TransformersBackendConfig = Field(
        description="Transformers backend used to extract token-level prompt embeddings."
    )
    reward_dataset_dir: str = Field(
        description="Directory produced by generate_reward_dataset."
    )
    train_embeddings_path: str | None = Field(
        default=None,
        description=(
            "Optional cached mean-pooled train prompt embeddings. Only used when "
            "reward_model.model_type=mean_pool."
        ),
    )
    test_embeddings_path: str | None = Field(
        default=None,
        description=(
            "Optional cached mean-pooled test prompt embeddings. Only used when "
            "reward_model.model_type=mean_pool."
        ),
    )
    do_cache_avg_embedding: bool = Field(
        default=False,
        description=(
            "When true and reward_model.model_type=mean_pool, cache averaged "
            "prompt embeddings in the Hydra run directory using the standard "
            "train/test prompt embedding artifact filenames when no explicit "
            "embedding cache path is provided."
        ),
    )
    embedding_batch_size: int = Field(
        default=8,
        gt=0,
        description="Prompt embedding extraction batch size.",
    )
    feature_config: FeatureExpansionConfig = Field(
        default_factory=FeatureExpansionConfig.unary,
        description="Placement feature expansion used by the reward model.",
    )
    reward_model: RewardModelConfig = Field(
        description="Reward-model architecture configuration.",
    )
    optimizer: RewardOptimizerConfig = Field(
        description="Optimizer configuration for reward-model training.",
    )
    num_epochs: int = Field(
        default=50,
        gt=0,
        description="Number of training epochs.",
    )
    batch_size: int = Field(
        default=256,
        gt=0,
        description="Mini-batch size for reward-model training.",
    )
    eval_batch_size: int = Field(
        default=512,
        gt=0,
        description="Mini-batch size for reward-model evaluation.",
    )
    log_every_n_steps: int = Field(
        default=100,
        gt=0,
        description="Emit training logs after this many optimizer steps.",
    )
    training_device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto",
        description="Device used to train and evaluate the reward model.",
    )
    seed: int = Field(default=42, description="Random seed.")
