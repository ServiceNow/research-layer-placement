# Implementation Guide

## Step 1: Create Directory Structure

For a new polymorphic component:

```
src/{package_name}/
├── {component}/
│   ├── __init__.py           # Discriminated union export
│   ├── base_{component}.py   # Base class + config (colocated)
│   ├── {impl1}.py            # Implementation + config (colocated)
│   └── {impl2}.py            # Implementation + config (colocated)
└── conf/
    └── {component}/
        ├── {impl1}.yaml      # Hydra defaults for impl1
        └── {impl2}.yaml      # Hydra defaults for impl2
```

## Step 2: Implement Base Class (Colocated with Config)

```python
# base_{component}.py
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Self
from pydantic import BaseModel, Field

class Base{Component}(ABC):
    """Abstract base class for {component} implementations."""

    def __init__(self, config: "Base{Component}Config"):
        self.config = config

    @abstractmethod
    async def main_method(self, params) -> ReturnType:
        """Main functionality."""
        pass

    async def __aenter__(self) -> Self:
        return self

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup resources."""
        pass


# Config defined AFTER implementation class
class Base{Component}Config(BaseModel):
    """Base configuration for all {component} implementations."""

    model_config = {"arbitrary_types_allowed": True}

    # Discriminator field (filled by subclasses)
    {component}_type: str = Field(description="Type identifier")

    # Common fields shared by all implementations
    common_field: str = Field(description="...")

    # Each subclass must define this
    _impl_class: ClassVar[type["Base{Component}"]]

    def create_{component}(self) -> "Base{Component}":
        """Smart constructor."""
        return self._impl_class(config=self)
```

## Step 3: Implement Concrete Class (Colocated with Config)

```python
# {impl}.py
from typing import Literal, ClassVar
from pydantic import Field
from .base_{component} import Base{Component}, Base{Component}Config


class {Impl}(Base{Component}):
    """Concrete implementation using {technology}."""

    def __init__(self, config: "{Impl}Config"):
        super().__init__(config)
        self.resource = self._initialize()

    def _initialize(self):
        """Initialize resources from config."""
        return SomeResource(self.config.specific_param)

    async def main_method(self, params) -> ReturnType:
        # Implementation
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # Cleanup
        pass


# Config defined AFTER implementation class
class {Impl}Config(Base{Component}Config):
    """Configuration for {Impl}."""

    _impl_class: ClassVar[type[Base{Component}]] = {Impl}

    # Discriminator with Literal type
    {component}_type: Literal["{impl_tag}"] = "{impl_tag}"

    # Implementation-specific fields
    specific_param: str = Field(description="...")
```

## Step 4: Create Discriminated Union

```python
# {component}/__init__.py
from typing import Annotated
from pydantic import Field

from .{impl1} import {Impl1}Config
from .{impl2} import {Impl2}Config

{Component}Config = Annotated[
    {Impl1}Config | {Impl2}Config,
    Field(discriminator="{component}_type"),
]
```

## Step 5: Create Hydra YAML Files

```yaml
# conf/{component}/{impl}.yaml
# @package {component}

{component}_type: "{impl_tag}"
common_field: "default_value"
specific_param: "impl_specific_default"
```

## Step 6: Wire into Root Config

```python
# config.py
from pydantic import BaseModel, Field
from .{component} import {Component}Config

class MainConfig(BaseModel):
    {component}: {Component}Config = Field(description="...")
```

## Step 7: Application Entry Point

```python
# __main__.py
import asyncio
import hydra
from omegaconf import DictConfig, OmegaConf

from .config import MainConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Convert to Pydantic (validates!)
    config = MainConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))

    # Use smart constructor
    component = config.{component}.create_{component}()

    asyncio.run(run(component))

if __name__ == "__main__":
    main()
```

---

## Checklist: Adding New Implementation

- [ ] Create `{impl}.py` with implementation class
- [ ] Add `{Impl}Config` with:
  - [ ] `_impl_class: ClassVar = {Impl}`
  - [ ] Discriminator: `{component}_type: Literal["{tag}"] = "{tag}"`
  - [ ] Implementation-specific fields
- [ ] Add to union in `__init__.py`
- [ ] Create `conf/{component}/{impl}.yaml`
- [ ] Update `pyproject.toml` if new dependencies (optional=true)
- [ ] Test: config validates, smart constructor works, Hydra override works

---

## Troubleshooting

**`ValidationError: Discriminator 'X' is missing`**
- Add discriminator field to YAML: `{component}_type: "{value}"`

**`TypeError: missing required argument 'config'`**
- Use smart constructor: `config.create_{component}()`

**Circular import**
- Use `TYPE_CHECKING` guard and string annotations

**`KeyError` accessing nested config**
- Check `defaults:` list and `@package` annotations in YAML
