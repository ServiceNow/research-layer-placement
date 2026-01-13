# Common Variations

## Nested Discriminated Unions

For multi-level polymorphism:

```python
# Top level: data sources
DataSourceConfig = Annotated[
    DatabaseConfig | FileConfig | APIConfig,
    Field(discriminator="source_type")
]

# Nested level: database types
DatabaseConfig = Annotated[
    PostgresConfig | MySQLConfig | MongoConfig,
    Field(discriminator="db_type")
]
```

---

## Shared Base Configuration

Extract common config to a shared base:

```python
class SharedConfig(BaseModel):
    timeout: float
    retries: int

class MainConfig(BaseModel):
    shared: SharedConfig
    component1: Component1Config
    component2: Component2Config
```

Reference in implementations:
```python
def __init__(self, config: Config, shared: SharedConfig):
    self.timeout = shared.timeout
```

---

## Synchronous Application

For non-async applications:

```python
def main(cfg: DictConfig) -> None:
    config = MainConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))

    component = config.component.create_component()
    with component:  # Sync context manager
        result = component.process(...)
```

---

## Lifecycle Hooks

Extend base classes with lifecycle hooks:

```python
class BaseComponent(ABC):
    async def initialize(self) -> None:
        """Called after __init__ but before first use."""
        pass

    async def shutdown(self) -> None:
        """Called before cleanup."""
        pass

    async def on_error(self, error: Exception) -> None:
        """Called when error occurs."""
        pass
```

---

## Custom Validation

Cross-field validation with Pydantic validators:

```python
from pydantic import model_validator
from typing import Self

class ComponentConfig(BaseComponentConfig):

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if self.param1 > self.param2:
            raise ValueError("param1 must be <= param2")
        return self
```

---

## State as Pydantic BaseModel

All stateful objects that need serialization must be Pydantic BaseModels:

```python
from pydantic import BaseModel

class State(BaseModel):
    """State that can be serialized."""
    model_config = {"frozen": True}  # Immutable if needed
    field1: str
    field2: int
    nested: dict[str, list[int]]

# Serialization
json_str = state.model_dump_json(indent=2)
dict_data = state.model_dump()

# Deserialization
loaded = State.model_validate_json(json_str)
loaded = State.model_validate(dict_data)
```

**Never** write custom serialization - always use Pydantic's built-in methods.

---

## Anti-Patterns

### Don't add defaults to Pydantic classes
```python
# BAD
class Config(BaseModel):
    param: int = 42
```
```yaml
# GOOD - put in YAML
param: 42
```

### Don't use string discriminators without Literal
```python
# BAD
backend_type: str = "mock"

# GOOD
backend_type: Literal["mock"] = "mock"
```

### Don't skip validation
```python
# BAD
return Thing(cfg["param"])

# GOOD
config = ThingConfig.model_validate(OmegaConf.to_container(cfg))
return config.create_thing()
```

### Don't maintain `__all__`
```python
# BAD - extra maintenance
__all__ = ["Thing1", "Thing2"]

# GOOD - just import what you need
from .module import Thing1, Thing2
```
