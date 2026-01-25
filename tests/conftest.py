"""Pytest configuration and fixtures for place_layers tests."""

from typing import Any, cast

import hydra
import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


@pytest.fixture
def hydra_config() -> dict[str, Any]:
    """Load the complete Hydra configuration from YAML files.

    This fixture initializes Hydra with the project's config directory,
    enabling config composition for tests.
    """
    # Clear any existing Hydra state
    GlobalHydra.instance().clear()

    with hydra.initialize(
        config_path="../src/place_layers/conf", version_base=None
    ):
        cfg = hydra.compose(config_name="config")
        return cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
