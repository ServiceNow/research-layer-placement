#!/usr/bin/env python3

import asyncio
import logging
import sys
import warnings

import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationResolutionError

from place_layers.command import command_adapter
from place_layers.command.base_command import CommandError


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    warnings.simplefilter("default")
    logging.captureWarnings(True)

    logger = logging.getLogger(__name__)

    try:
        # Parse config to discriminated union type
        config = command_adapter.validate_python(
            OmegaConf.to_container(cfg, resolve=True)
        )

        # Create and run command
        command = config.create_command()
        asyncio.run(command.run())

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except InterpolationResolutionError as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)
    except CommandError as e:
        logger.error("Command error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
