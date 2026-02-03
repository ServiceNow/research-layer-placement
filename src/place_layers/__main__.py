#!/usr/bin/env python3

import asyncio
import logging
import sys
import warnings

import hydra
import hashlib
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationResolutionError

from place_layers.command.base_command import CommandError


def _path_sanitize_fragment(value: object) -> str:
    s = "" if value is None else str(value)
    # Prevent accidental directory splitting when override values contain path separators
    # (e.g. URLs like http://... or Windows paths with backslashes).
    return (
        s.replace("\\", "__")
        .replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
        .replace('"', "")
        .replace("'", "")
    )


OmegaConf.register_new_resolver(
    "path_sanitize_fragment", _path_sanitize_fragment, replace=True
)


def _path_hash(value: object, length: object = 12) -> str:
    s = "" if value is None else str(value)
    try:
        n = int(length)
    except Exception:
        n = 12
    n = max(4, min(n, 40))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def _path_truncate(value: object, max_len: object = 120) -> str:
    s = "" if value is None else str(value)
    try:
        n = int(max_len)
    except Exception:
        n = 120
    if n <= 0:
        return ""
    return s[:n]


OmegaConf.register_new_resolver("path_hash", _path_hash, replace=True)
OmegaConf.register_new_resolver("path_truncate", _path_truncate, replace=True)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    warnings.simplefilter("default")
    logging.captureWarnings(True)

    logger = logging.getLogger(__name__)

    try:
        from place_layers.command import command_adapter

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
