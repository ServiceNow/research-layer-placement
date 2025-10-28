"""Pytest configuration and fixtures for the benchmarking test suite."""

from typing import Any

import hydra
from omegaconf import OmegaConf
import pytest

from place_layers.config import MainConfig

@pytest.fixture
def hydra_config() -> dict[str, Any]:
    """Load the complete Hydra configuration from YAML files."""
    with hydra.initialize(
        config_path="../src/place_layers/conf", version_base=None
    ):
        cfg = hydra.compose(config_name="config")
        return OmegaConf.to_container(cfg, resolve=True)

@pytest.fixture
def main_config(hydra_config: dict[str, Any]) -> MainConfig:
    """Main configuration loaded from Hydra defaults."""
    return MainConfig.model_validate(hydra_config)
