# Core Architectural Patterns

## Pattern 1: Configuration Separation

**Problem**: Hardcoded defaults mixed with business logic, configuration scattered across codebase.

**Solution**: Strict separation of concerns.

```
Hydra YAML Files        →  All default values (infrastructure)
Pydantic BaseModel      →  Schemas, types, validation (code)
Application Code        →  Business logic only (no config)
```

**Key Principle**: **ZERO defaults in Pydantic classes**. All defaults externalized to YAML.

**Benefits**:
- Change defaults without touching code
- Different defaults per environment (dev/prod)
- Version control config separately from code
- Non-programmers can modify configurations

---

## Pattern 2: Discriminated Unions (Tagged Sums)

**Problem**: Need to configure different implementations of the same interface.

**Solution**: Use Pydantic's discriminated unions with a type tag.

```python
from typing import Annotated, Literal
from pydantic import BaseModel, Field

# Define union with discriminator
ComponentConfig = Annotated[
    Implementation1Config
    | Implementation2Config
    | Implementation3Config,
    Field(discriminator="component_type")
]

# Each implementation has Literal type for discriminator
class Implementation1Config(BaseComponentConfig):
    component_type: Literal["impl1"] = "impl1"
    # impl1-specific fields...
```

**Benefits**:
- Type-safe polymorphism
- IDE autocomplete for each implementation's specific fields
- Impossible to create invalid configurations
- Clear error messages when discriminator is wrong

**When to use**:
- Multiple implementations of an interface/protocol
- Plugin systems
- Strategy pattern
- Backend/handler/processor abstractions

---

## Pattern 3: Smart Constructors

**Problem**: Need to instantiate the correct implementation class from configuration.

**Solution**: Config classes carry metadata about their implementation and provide a factory method.

```python
from typing import TYPE_CHECKING, ClassVar
from pydantic import BaseModel

if TYPE_CHECKING:
    from .base_component import BaseComponent

class BaseComponentConfig(BaseModel):
    # Class-level metadata (not instance data)
    _impl_class: ClassVar[type["BaseComponent"]]

    def create_component(self) -> "BaseComponent":
        """Smart constructor: config → implementation"""
        return self._impl_class(config=self)
```

**Benefits**:
- Single source of truth for config → impl mapping
- No switch statements or if/elif chains
- Easy to add new implementations
- Config is immutable and passed to implementation

**When to use**:
- Whenever you have discriminated unions
- Factory pattern scenarios
- Dependency injection

---

## How the Patterns Work Together

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Entry Point                  │
│  @hydra.main() → DictConfig → Pydantic Validation          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Pattern 1: Configuration Separation            │
│  • Hydra YAML files: Store all default values              │
│  • Pydantic Models: Define schemas and validation          │
│  • Zero defaults in code                                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│           Pattern 2: Discriminated Unions                   │
│  • Polymorphic configuration with type safety               │
│  • Discriminator field selects implementation               │
│  • Literal types ensure correctness                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│             Pattern 3: Smart Constructors                   │
│  • Config classes know their implementation class           │
│  • Factory method instantiates correct implementation       │
│  • Config is copied into instance                           │
└─────────────────────────────────────────────────────────────┘
```

---

## When to Use These Patterns

### Good Fit

- **Multiple implementations** of the same abstraction
- **Complex configuration** with many parameters
- **Runtime flexibility** - switch implementations via config
- **Type safety requirements** - compile-time checking
- **Plugin architecture** - extensible systems

### Not Recommended

- Simple script with < 10 configuration options
- Configuration never changes
- Only one implementation exists
- Startup time is critical (Pydantic adds overhead)
