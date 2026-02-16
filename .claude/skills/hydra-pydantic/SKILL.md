---
name: hydra-pydantic
description: Use for ANY task that adds, modifies, or touches commands, features, components, Pydantic models, Hydra YAML configs, or configuration of any kind. In this codebase every new feature and every command change involves Hydra+Pydantic configuration, so this skill applies to virtually all implementation work. Covers: creating or modifying BaseModel config classes, editing YAML config files, adding config fields, creating config groups, implementing polymorphic components (discriminated unions, smart constructors), adding new commands, and refactoring existing ones. Critical rules enforced: NEVER use dataclasses for configs (always Pydantic BaseModel), NEVER put defaults or default_factory in Pydantic models (all defaults live in YAML), always use Literal discriminators for unions.
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

- Adding or modifying ANY command or feature (all commands have configs)
- Creating or editing Pydantic BaseModel classes
- Editing or creating Hydra YAML config files
- Adding fields to existing config classes
- Creating new config groups
- Implementing new polymorphic component types
- Debugging discriminated union or smart constructor issues
- Any task where configuration structure decisions are being made

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
