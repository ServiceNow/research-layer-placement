"""Evaluate command for running evaluation without search.

This command evaluates one or more placements using a configured evaluator,
without running any optimization search. Useful for:
- Testing evaluator setup
- Evaluating baseline placements (e.g., all-attention)
- Comparing specific placements manually
- Validating search results

Usage
-----
    python -m place_layers command=evaluate evaluator=generative \\
        placements='[{name: all_attention, layers: [attention, attention, ...]}]'
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Literal

from hydra.core.hydra_config import HydraConfig
from pydantic import BaseModel, Field, model_validator

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.core.types import LayerPlacement
from place_layers.evaluator import EvaluatorConfig

logger = logging.getLogger(__name__)


class NamedPlacement(BaseModel):
    """A placement with a name for identification."""

    model_config = {"frozen": True}

    name: str = Field(description="Name/identifier for this placement")
    layers: list[str] = Field(
        description="Mixer name for each layer. If shorter than num_layers, "
        "repeats cyclically (periodic boundary conditions)"
    )


class EvaluationResult(BaseModel):
    """Result of evaluating a single placement."""

    model_config = {"frozen": True}

    name: str = Field(description="Placement name")
    placement: list[str] = Field(description="Mixer names per layer")
    score: float = Field(description="Evaluation score")
    placement_indices: list[int] = Field(description="Mixer indices per layer")


class EvaluateCommandOutput(BaseModel):
    """Output of the evaluate command."""

    model_config = {"frozen": True}

    timestamp: str = Field(description="ISO timestamp of evaluation")
    evaluator_type: str = Field(description="Type of evaluator used")
    mixer_names: list[str] = Field(description="Available mixer types")
    num_layers: int = Field(description="Number of layers")
    results: list[EvaluationResult] = Field(description="Evaluation results")


class EvaluateCommand(BaseCommand):
    """Run evaluation on specific placements without search."""

    config: "EvaluateCommandConfig"

    async def run(self) -> None:
        """Execute evaluation."""
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)

        logger.info(f"Output directory: {output_dir}")
        logger.info("Starting evaluation...")

        # Create and use evaluator with context manager for proper cleanup
        async with self.config.evaluator.create_evaluator() as evaluator:
            num_layers = evaluator.num_layers
            num_mixers = evaluator.num_mixers

            logger.info(
                f"Evaluator: {self.config.evaluator.evaluator_type}, "
                f"{num_layers} layers, {num_mixers} mixers"
            )

            # Validate mixer_names length
            if len(self.config.mixer_names) != num_mixers:
                raise ValueError(
                    f"mixer_names has {len(self.config.mixer_names)} entries, "
                    f"but evaluator has {num_mixers} mixers"
                )

            # Build name-to-index mapping
            name_to_idx = {name: idx for idx, name in enumerate(self.config.mixer_names)}

            logger.info(f"Evaluating {len(self.config.placements)} placement(s)...")

            # Evaluate each placement
            results: list[EvaluationResult] = []

            for placement_spec in self.config.placements:
                # Extend layers cyclically if shorter than num_layers (periodic BC)
                spec_layers = placement_spec.layers
                if len(spec_layers) < num_layers:
                    extended_layers = [
                        spec_layers[i % len(spec_layers)] for i in range(num_layers)
                    ]
                    logger.info(
                        f"  '{placement_spec.name}': extended {len(spec_layers)} layers "
                        f"to {num_layers} (periodic)"
                    )
                elif len(spec_layers) > num_layers:
                    raise ValueError(
                        f"Placement '{placement_spec.name}' has {len(spec_layers)} layers, "
                        f"but model only has {num_layers} layers"
                    )
                else:
                    extended_layers = list(spec_layers)

                # Convert mixer names to indices
                try:
                    indices = tuple(
                        name_to_idx[name] for name in extended_layers
                    )
                except KeyError as e:
                    raise ValueError(
                        f"Unknown mixer name in placement '{placement_spec.name}': {e}. "
                        f"Valid names: {list(name_to_idx.keys())}"
                    ) from e

                # Create LayerPlacement
                layer_placement = LayerPlacement(assignments=indices)

                # Evaluate
                logger.info(f"Evaluating '{placement_spec.name}'...")
                score = await evaluator.evaluate(layer_placement)

                result = EvaluationResult(
                    name=placement_spec.name,
                    placement=extended_layers,
                    score=score,
                    placement_indices=list(indices),
                )
                results.append(result)

                logger.info(f"  '{placement_spec.name}': score={score:.6f}")

        # Log summary
        logger.info("=" * 60)
        logger.info("Evaluation Results:")
        for result in sorted(results, key=lambda r: -r.score):
            logger.info(f"  {result.name}: {result.score:.6f}")
        logger.info("=" * 60)

        # Save results
        output = EvaluateCommandOutput(
            timestamp=datetime.now().isoformat(),
            evaluator_type=self.config.evaluator.evaluator_type,
            mixer_names=self.config.mixer_names,
            num_layers=num_layers,
            results=results,
        )

        output_path = output_dir / "evaluation_results.json"
        with open(output_path, "w") as f:
            f.write(output.model_dump_json(indent=2))

        logger.info(f"Results saved to: {output_path}")
        logger.info("Evaluation complete!")


class EvaluateCommandConfig(BaseCommandConfig):
    """Configuration for evaluate command."""

    _command_class: ClassVar[type[BaseCommand]] = EvaluateCommand

    command: Literal["evaluate"] = "evaluate"

    # Evaluator configuration (discriminated union)
    evaluator: EvaluatorConfig = Field(description="Evaluator configuration")

    # Mixer names (defines name-to-index mapping)
    mixer_names: list[str] = Field(
        description="List of mixer names in index order "
        "(e.g., ['attention', 'mamba', 'gdn'])"
    )

    # Placements to evaluate
    placements: list[NamedPlacement] = Field(
        description="List of placements to evaluate"
    )

    @model_validator(mode="after")
    def validate_config(self) -> "EvaluateCommandConfig":
        """Validate configuration."""
        if not self.placements:
            raise ValueError("At least one placement must be specified")

        if not self.mixer_names:
            raise ValueError("mixer_names must not be empty")

        return self
