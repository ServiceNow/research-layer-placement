"""Evaluate command for running evaluation without search.

This command evaluates a single placement using a configured evaluator,
without running any optimization search. Useful for:
- Testing evaluator setup
- Evaluating baseline placements (e.g., all-attention)
- Comparing specific placements (use Hydra --multirun for sweeps)

Usage
-----
    # Single placement
    python -m place_layers command=evaluate placement=all_attention

    # Sweep over placements
    python -m place_layers command=evaluate placement=all_attention,every_4th_gdn --multirun

    # No placement (for non-supernet models)
    python -m place_layers command=evaluate placement=null
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Literal

from hydra.core.hydra_config import HydraConfig
from pydantic import BaseModel, Field

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.core.types import LayerPlacement
from place_layers.evaluator import EvaluatorConfig

logger = logging.getLogger(__name__)


class EvaluationResult(BaseModel):
    """Result of evaluating a placement."""

    model_config = {"frozen": True}

    placement: list[str] | None = Field(
        description="Mixer names per layer (null if no placement)"
    )
    score: float = Field(description="Evaluation score")
    placement_indices: list[int] | None = Field(
        description="Mixer indices per layer (null if no placement)"
    )


class EvaluateCommandOutput(BaseModel):
    """Output of the evaluate command."""

    model_config = {"frozen": True}

    timestamp: str = Field(description="ISO timestamp of evaluation")
    evaluator_type: str = Field(description="Type of evaluator used")
    mixer_names: list[str] = Field(description="Available mixer types")
    num_layers: int = Field(description="Number of layers")
    result: EvaluationResult = Field(description="Evaluation result")


class EvaluateCommand(BaseCommand):
    """Run evaluation on a single placement without search."""

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
            mixer_names = evaluator.mixer_names

            logger.info(
                f"Evaluator: {self.config.evaluator.evaluator_type}, "
                f"{num_layers} layers, {num_mixers} mixers"
            )
            logger.info(f"Available mixers: {mixer_names}")

            # Build name-to-index mapping
            name_to_idx = {name: idx for idx, name in enumerate(mixer_names)}

            # Check if model supports placement (multiple mixer types)
            supports_placement = len(set(mixer_names)) > 1

            # Handle placement
            if self.config.placement is not None:
                # Placement specified - model must support it
                if not supports_placement:
                    raise ValueError(
                        f"Placement specified but model only has one mixer type "
                        f"({mixer_names[0]}). Use placement=null for "
                        f"non-supernet models."
                    )

                # Extend layers cyclically if shorter than num_layers (periodic BC)
                spec_layers = self.config.placement
                if len(spec_layers) < num_layers:
                    extended_layers = [
                        spec_layers[i % len(spec_layers)] for i in range(num_layers)
                    ]
                    logger.info(
                        f"Extended {len(spec_layers)} layers to {num_layers} (periodic BC)"
                    )
                elif len(spec_layers) > num_layers:
                    raise ValueError(
                        f"Placement has {len(spec_layers)} layers, "
                        f"but model only has {num_layers} layers"
                    )
                else:
                    extended_layers = list(spec_layers)

                # Convert mixer names to indices
                try:
                    indices = tuple(name_to_idx[name] for name in extended_layers)
                except KeyError as e:
                    raise ValueError(
                        f"Unknown mixer name in placement: {e}. "
                        f"Valid names: {list(name_to_idx.keys())}"
                    ) from e

                # Create LayerPlacement and evaluate
                layer_placement = LayerPlacement(assignments=indices)
                logger.info(f"Evaluating placement: {extended_layers[:6]}...")
                score = await evaluator.evaluate(layer_placement)

                result = EvaluationResult(
                    placement=extended_layers,
                    score=score,
                    placement_indices=list(indices),
                )
            else:
                # No placement - evaluate with default model configuration
                logger.info("No placement specified, evaluating default configuration")

                # Use all-zeros placement (first mixer type for all layers)
                default_indices = tuple(0 for _ in range(num_layers))
                layer_placement = LayerPlacement(assignments=default_indices)
                score = await evaluator.evaluate(layer_placement)

                result = EvaluationResult(
                    placement=None,
                    score=score,
                    placement_indices=None,
                )

            logger.info(f"Score: {score:.6f}")

        # Save results
        output = EvaluateCommandOutput(
            timestamp=datetime.now().isoformat(),
            evaluator_type=self.config.evaluator.evaluator_type,
            mixer_names=list(mixer_names),
            num_layers=num_layers,
            result=result,
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
    # The evaluator provides model structure (num_layers, mixer_names) after initialization
    evaluator: EvaluatorConfig = Field(description="Evaluator configuration")

    # Single placement (use Hydra --multirun for sweeps)
    placement: list[str] | None = Field(
        default=None,
        description="Layer placement as list of mixer names. Extended cyclically "
        "if shorter than num_layers (periodic BC). Use null for non-supernet models."
    )
