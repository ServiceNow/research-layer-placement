"""Explore placements independently for each traced problem."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar, Literal

from hydra.core.hydra_config import HydraConfig

from place_layers.command.explore_placements import (
    ExplorePlacementsCommand,
    ExplorePlacementsCommandConfig,
)
from place_layers.core.trace_problem_split import build_per_problem_evaluators


logger = logging.getLogger(__name__)


class ExplorePerProblemCommand(ExplorePlacementsCommand):
    """Build one exploration dataset per traced problem."""

    config: "ExplorePerProblemCommandConfig"

    async def run(self) -> None:
        """Execute adaptive exploration independently for each problem."""
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)

        logger.info("Output directory: %s", output_dir)
        logger.info("Starting per-problem placement exploration...")

        derived_evaluators, manifest = build_per_problem_evaluators(
            self.config.evaluators,
            output_dir / "_problem_traces",
        )
        (output_dir / "per_problem_manifest.json").write_text(
            manifest.model_dump_json(indent=2)
        )

        from contextlib import nullcontext

        backend_cm = (
            self.config.backend.create_backend()
            if self.config.backend
            else nullcontext()
        )
        async with backend_cm as backend:
            async with derived_evaluators.create_evaluators(
                output_dir, backend=backend
            ) as evaluator_instances:
                cost_model = self.config.cost_model.create_cost_model()
                num_layers = evaluator_instances.num_layers

                for obj_name in evaluator_instances.objective_names:
                    await self._explore_objective(
                        evaluator_instances=evaluator_instances,
                        obj_name=obj_name,
                        cost_model=cost_model,
                        num_layers=num_layers,
                        output_dir=output_dir,
                    )


class ExplorePerProblemCommandConfig(ExplorePlacementsCommandConfig):
    """Configuration for explore_per_problem command."""

    _impl_class: ClassVar[type[ExplorePlacementsCommand]] = ExplorePerProblemCommand

    command: Literal["explore_per_problem"] = "explore_per_problem"
