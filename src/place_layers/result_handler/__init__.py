"""Result handler implementations."""

from typing import Annotated

from pydantic import Field

from place_layers.result_handler.json_result_handler import JSONResultHandlerConfig
from place_layers.result_handler.yaml_result_handler import YAMLResultHandlerConfig


# Discriminated union of all result handlers
ResultHandlerConfig = Annotated[
    JSONResultHandlerConfig | YAMLResultHandlerConfig,
    Field(discriminator="handler_type"),
]
