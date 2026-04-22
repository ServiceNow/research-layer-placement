"""Generate a prompt-placement reward dataset from teacher traces."""

from __future__ import annotations

import logging
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from hydra.core.hydra_config import HydraConfig
from pydantic import BaseModel, Field, model_validator

from place_layers.backend import BackendConfig
from place_layers.backend.types import SupernetStructure
from place_layers.command.base_command import BaseCommand, CommandError
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.cost_model import CostModelConfig
from place_layers.router import (
    load_prompt_trace_groups,
    sample_random_placements,
    save_sampled_placements,
    score_problem_trace_matrix,
)

logger = logging.getLogger(__name__)


class RewardDatasetSummary(BaseModel):
    """Summary artifact from generate_reward_dataset."""

    num_train_prompts: int
    num_test_prompts: int
    num_placements: int
    num_test_new_placements: int
    num_layers: int
    mixer_names: list[str]
    teacher_mixer_name: str
    train_prompts_train_placements_observed_fraction: float
    train_prompts_test_placements_observed_fraction: float
    test_prompts_train_placements_observed_fraction: float
    test_prompts_test_placements_observed_fraction: float


def _ensure_sampled_placements(
    *,
    path: Path,
    placements: list[tuple[str, ...]],
) -> None:
    """Persist sampled placements early and validate them on resume."""
    expected = [list(placement) for placement in placements]
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != expected:
            raise CommandError(
                f"Existing sampled placements at {path} do not match the current run"
            )
        return
    save_sampled_placements(path=path, placements=placements)


class GenerateRewardDatasetCommand(BaseCommand):
    """Sample placements and score prompt-level rewards for train/test splits."""

    config: "GenerateRewardDatasetCommandConfig"

    async def run(self) -> None:
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        placements_path = output_dir / "sampled_placements.json"
        test_new_placements_path = output_dir / "test_sampled_placements.json"
        train_train_scores_jsonl_path = (
            output_dir / "train_prompts_train_placements_scores.jsonl"
        )
        train_test_scores_jsonl_path = (
            output_dir / "train_prompts_test_placements_scores.jsonl"
        )
        test_train_scores_jsonl_path = (
            output_dir / "test_prompts_train_placements_scores.jsonl"
        )
        test_test_scores_jsonl_path = (
            output_dir / "test_prompts_test_placements_scores.jsonl"
        )
        summary_path = output_dir / "reward_dataset_summary.json"

        train_problems = load_prompt_trace_groups(Path(self.config.train_trace_dir))
        test_problems = load_prompt_trace_groups(Path(self.config.test_trace_dir))
        if not train_problems:
            raise CommandError("Training trace set is empty")
        if not test_problems:
            raise CommandError("Test trace set is empty")

        backend_cm = (
            self.config.backend.create_backend()
            if self.config.backend is not None
            else nullcontext()
        )
        async with backend_cm as backend:
            if backend is None:
                raise CommandError("generate_reward_dataset requires a backend")

            structure = backend.model_structure
            if not isinstance(structure, SupernetStructure):
                raise CommandError("generate_reward_dataset requires a supernet backend")

            num_layers = structure.num_layers
            mixer_names = list(structure.mixer_names)
            if self.config.teacher_mixer_name not in mixer_names:
                raise CommandError(
                    f"Teacher mixer '{self.config.teacher_mixer_name}' "
                    f"is not available in {mixer_names}"
                )
            cost_model = None
            if self.config.placement_sampling == "cost_stratified":
                cost_model = self.config.cost_model.create_cost_model()
                cost_mixers = set(cost_model.phase_order)
                if cost_mixers != set(mixer_names):
                    raise CommandError(
                        "Cost model mixers do not match backend mixers: "
                        f"backend={mixer_names}, cost_model={list(cost_model.phase_order)}"
                    )

            placements = sample_random_placements(
                mixer_names=mixer_names,
                num_layers=num_layers,
                num_placements=self.config.num_placements,
                seed=self.config.seed,
                placement_sampling=self.config.placement_sampling,
                cost_model=cost_model,
                include_teacher=self.config.include_teacher_placement,
                teacher_mixer_name=self.config.teacher_mixer_name,
            )
            _ensure_sampled_placements(path=placements_path, placements=placements)

            test_new_placements = sample_random_placements(
                mixer_names=mixer_names,
                num_layers=num_layers,
                num_placements=self.config.num_test_placements,
                seed=self.config.seed + 1,
                placement_sampling=self.config.placement_sampling,
                cost_model=cost_model,
                exclude_placements=placements,
                include_teacher=False,
                teacher_mixer_name=self.config.teacher_mixer_name,
            )
            _ensure_sampled_placements(
                path=test_new_placements_path,
                placements=test_new_placements,
            )

            (
                train_train_scores,
                train_train_observed_mask,
            ) = await score_problem_trace_matrix(
                backend=backend,
                problems=train_problems,
                placements=placements,
                progress_description="Train prompts / train placements reward scoring",
                progress_jsonl_path=train_train_scores_jsonl_path,
                max_problems_per_placement=self.config.max_problems_per_placement,
                seed=self.config.seed,
            )
            (
                train_test_scores,
                train_test_observed_mask,
            ) = await score_problem_trace_matrix(
                backend=backend,
                problems=train_problems,
                placements=test_new_placements,
                progress_description="Train prompts / test placements reward scoring",
                progress_jsonl_path=train_test_scores_jsonl_path,
                max_problems_per_placement=self.config.max_problems_per_placement,
                seed=self.config.seed + 1,
            )
            (
                test_train_scores,
                test_train_observed_mask,
            ) = await score_problem_trace_matrix(
                backend=backend,
                problems=test_problems,
                placements=placements,
                progress_description="Test prompts / train placements reward scoring",
                progress_jsonl_path=test_train_scores_jsonl_path,
                max_problems_per_placement=self.config.max_problems_per_placement,
                seed=self.config.seed,
            )
            (
                test_test_scores,
                test_test_observed_mask,
            ) = await score_problem_trace_matrix(
                backend=backend,
                problems=test_problems,
                placements=test_new_placements,
                progress_description="Test prompts / test placements reward scoring",
                progress_jsonl_path=test_test_scores_jsonl_path,
                max_problems_per_placement=self.config.max_problems_per_placement,
                seed=self.config.seed + 1,
            )

        summary = RewardDatasetSummary(
            num_train_prompts=len(train_problems),
            num_test_prompts=len(test_problems),
            num_placements=len(placements),
            num_test_new_placements=len(test_new_placements),
            num_layers=num_layers,
            mixer_names=mixer_names,
            teacher_mixer_name=self.config.teacher_mixer_name,
            train_prompts_train_placements_observed_fraction=float(
                np.mean(train_train_observed_mask)
            ),
            train_prompts_test_placements_observed_fraction=float(
                np.mean(train_test_observed_mask)
            ),
            test_prompts_train_placements_observed_fraction=float(
                np.mean(test_train_observed_mask)
            ),
            test_prompts_test_placements_observed_fraction=float(
                np.mean(test_test_observed_mask)
            ),
        )
        summary_path.write_text(summary.model_dump_json(indent=2))

        logger.info("Saved sampled placements to %s", placements_path)
        logger.info("Saved new test placements to %s", test_new_placements_path)
        logger.info("Saved reward dataset summary to %s", summary_path)


class GenerateRewardDatasetCommandConfig(BaseCommandConfig):
    """Configuration for generate_reward_dataset."""

    _impl_class: ClassVar[type[BaseCommand]] = GenerateRewardDatasetCommand

    command: Literal["generate_reward_dataset"] = "generate_reward_dataset"

    @model_validator(mode="before")
    @classmethod
    def _validate_cost_model_requirement(cls, data: Any) -> Any:
        if isinstance(data, dict):
            placement_sampling = data.get("placement_sampling", "local")
            if placement_sampling == "cost_stratified" and data.get("cost_model") is None:
                raise ValueError(
                    "cost_model must be provided when placement_sampling='cost_stratified'"
                )
        return data

    backend: BackendConfig = Field(
        description="Backend used to score teacher traces under sampled placements."
    )
    cost_model: CostModelConfig | None = Field(
        default=None,
        description="Cost model used for cost_stratified placement sampling.",
    )
    train_trace_dir: str = Field(
        description="Directory containing precompute_traces output for training prompts."
    )
    test_trace_dir: str = Field(
        description="Directory containing precompute_traces output for test prompts."
    )
    num_placements: int = Field(
        default=32,
        gt=0,
        description="Number of random placements to score on each trace split.",
    )
    num_test_placements: int = Field(
        default=32,
        gt=0,
        description="Number of fresh test-only placements to score on the test split.",
    )
    seed: int = Field(default=42, description="Random seed.")
    placement_sampling: Literal["local", "cost_stratified"] = Field(
        default="cost_stratified",
        description="Placement sampler. 'local' samples each layer i.i.d.; "
        "'cost_stratified' samples allocations uniformly over rounded cost levels.",
    )
    max_problems_per_placement: int | None = Field(
        default=None,
        description="Optional prompt subsample per placement for cheaper dataset generation.",
    )
    teacher_mixer_name: str = Field(
        default="attention",
        description="Mixer name used for the teacher/all-teacher placement.",
    )
    include_teacher_placement: bool = Field(
        default=True,
        description="Whether to include the all-teacher placement among sampled placements.",
    )
