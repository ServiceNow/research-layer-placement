"""Result handler implementations."""

from typing import Annotated

from pydantic import Field

from place_layers.result_handler.json_result_handler import JSONResultHandlerConfig

# Discriminated union of all result handlers
ResultHandlerConfig = Annotated[
    JSONResultHandlerConfig,
    Field(discriminator="handler_type"),
]
