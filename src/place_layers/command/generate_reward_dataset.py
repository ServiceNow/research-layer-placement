"""Generate a prompt-placement reward dataset from teacher traces."""

from __future__ import annotations

import logging
import json
from contextlib import nullcontext
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from hydra.core.hydra_config import HydraConfig
from pydantic import BaseModel, Field

from place_layers.backend import BackendConfig
from place_layers.backend.types import SupernetStructure
from place_layers.command.base_command import BaseCommand, CommandError
from place_layers.command.base_command_config import BaseCommandConfig
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
    num_layers: int
    mixer_names: list[str]
    teacher_mixer_name: str
    train_observed_fraction: float
    test_observed_fraction: float


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
        train_scores_jsonl_path = output_dir / "train_placement_scores.jsonl"
        test_scores_jsonl_path = output_dir / "test_placement_scores.jsonl"
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

            placements = sample_random_placements(
                mixer_names=mixer_names,
                num_layers=num_layers,
                num_placements=self.config.num_placements,
                seed=self.config.seed,
                include_teacher=self.config.include_teacher_placement,
                teacher_mixer_name=self.config.teacher_mixer_name,
            )
            _ensure_sampled_placements(path=placements_path, placements=placements)

            train_scores, train_observed_mask = await score_problem_trace_matrix(
                backend=backend,
                problems=train_problems,
                placements=placements,
                progress_description="Train reward scoring",
                progress_jsonl_path=train_scores_jsonl_path,
                max_problems_per_placement=self.config.max_problems_per_placement,
                seed=self.config.seed,
            )
            test_scores, test_observed_mask = await score_problem_trace_matrix(
                backend=backend,
                problems=test_problems,
                placements=placements,
                progress_description="Test reward scoring",
                progress_jsonl_path=test_scores_jsonl_path,
                max_problems_per_placement=self.config.max_problems_per_placement,
                seed=self.config.seed,
            )

        summary = RewardDatasetSummary(
            num_train_prompts=len(train_problems),
            num_test_prompts=len(test_problems),
            num_placements=len(placements),
            num_layers=num_layers,
            mixer_names=mixer_names,
            teacher_mixer_name=self.config.teacher_mixer_name,
            train_observed_fraction=float(np.mean(train_observed_mask)),
            test_observed_fraction=float(np.mean(test_observed_mask)),
        )
        summary_path.write_text(summary.model_dump_json(indent=2))

        logger.info("Saved sampled placements to %s", placements_path)
        logger.info("Saved reward dataset summary to %s", summary_path)


class GenerateRewardDatasetCommandConfig(BaseCommandConfig):
    """Configuration for generate_reward_dataset."""

    _impl_class: ClassVar[type[BaseCommand]] = GenerateRewardDatasetCommand

    command: Literal["generate_reward_dataset"] = "generate_reward_dataset"

    backend: BackendConfig = Field(
        description="Backend used to score teacher traces under sampled placements."
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
    seed: int = Field(default=42, description="Random seed.")
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
