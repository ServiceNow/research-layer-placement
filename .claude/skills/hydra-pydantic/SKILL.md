---
name: hydra-pydantic
description: Use when adding new polymorphic components (backends, evaluators, runners), creating discriminated unions, implementing smart constructors, or setting up Hydra+Pydantic configuration patterns. Triggers on tasks like "add a new backend", "create new evaluator type", "implement new runner".
---

# Hydra + Pydantic Configuration Patterns

Reusable architectural patterns for building type-safe, configurable applications using Hydra + Pydantic.

## Quick Reference: Three Core Patterns

### Pattern 1: Configuration Separation
- **Hydra YAML files** → All default values (infrastructure)
- **Pydantic BaseModel** → Schemas, types, validation (code)
- **Key Principle**: ZERO defaults in Pydantic classes

### Pattern 2: Discriminated Unions
```python
ComponentConfig = Annotated[
    Impl1Config | Impl2Config | Impl3Config,
    Field(discriminator="component_type")
]

class Impl1Config(BaseComponentConfig):
    component_type: Literal["impl1"] = "impl1"
```

### Pattern 3: Smart Constructors
```python
class BaseComponentConfig(BaseModel):
    _impl_class: ClassVar[type["BaseComponent"]]

    def create_component(self) -> "BaseComponent":
        return self._impl_class(config=self)
```

## When to Use This Skill

- Adding a new backend/evaluator/runner implementation
- Creating a new polymorphic component type
- Setting up config for a new module with multiple implementations
- Debugging discriminated union or smart constructor issues

## Files to Create for New Component

```
{component}/base_{component}.py     # Base class + config (colocated)
{component}/{impl}.py               # Concrete impl + config (colocated)
{component}/__init__.py             # Discriminated union export
conf/{component}/{impl}.yaml        # Hydra defaults
```

## Detailed Documentation

- [Core Patterns](patterns.md) - Detailed explanation of the 3 patterns
- [Implementation Guide](implementation.md) - Step-by-step instructions
- [Variations](variations.md) - Common variations and extensions

## Anti-Patterns (Quick Reference)

| Don't | Do |
|-------|-----|
| `param: int = 42` in Pydantic | Put defaults in YAML |
| `backend_type: str = "mock"` | `backend_type: Literal["mock"]` |
| Skip Pydantic validation | Always use `model_validate()` |
| Separate config files | Colocate config with implementation |
