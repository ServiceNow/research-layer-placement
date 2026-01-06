# Configuration Architecture Pattern Template

This template describes reusable architectural patterns for building type-safe, configurable applications using Hydra + Pydantic. These patterns are domain-agnostic and can be applied to any Python project.

---

## Table of Contents

1. [When to Use These Patterns](#when-to-use-these-patterns)
2. [Pattern Overview](#pattern-overview)
3. [Core Architectural Patterns](#core-architectural-patterns)
4. [Implementation Guide](#implementation-guide)
5. [Boilerplate Templates](#boilerplate-templates)
6. [Common Variations](#common-variations)
7. [Example: This Codebase](#example-this-codebase)
8. [Hydra Config Directory Organization](#hydra-config-directory-organization)

---

## When to Use These Patterns

### ✅ Good Fit

Use this architecture when your project has:

- **Multiple implementations** of the same abstraction (backends, handlers, processors, etc.)
- **Complex configuration** with many parameters and nested structures
- **Runtime flexibility** - need to switch implementations via config files
- **Type safety requirements** - want compile-time checking of configurations
- **External configuration** - prefer YAML/JSON over hardcoded defaults
- **Plugin architecture** - extensible system where new implementations can be added
- **Multi-environment deployments** - different configs for dev/staging/prod

### ❌ Not Recommended

Don't use this architecture if:

- Simple script with < 10 configuration options
- Configuration never changes (hardcoded is fine)
- Only one implementation exists (no polymorphism needed)
- Startup time is critical (Pydantic validation adds overhead)
- Team unfamiliar with Hydra and Pydantic (learning curve exists)

---

## Pattern Overview

This architecture uses **three core patterns** that work together:

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

## Core Architectural Patterns

### Pattern 1: Configuration Separation

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

### Pattern 2: Discriminated Unions (Tagged Sums)

**Problem**: Need to configure different implementations of the same interface (e.g., different database backends, different auth providers).

**Solution**: Use Pydantic's discriminated unions with a type tag.

**Structure**:
```python
# Define union with discriminator
ComponentConfig = Annotated[
    Implementation1Config
    | Implementation2Config
    | Implementation3Config,
    Field(discriminator="{discriminator_field}")
]

# Each implementation has Literal type for discriminator
class Implementation1Config(BaseComponentConfig):
    {discriminator_field}: Literal["{impl1_tag}"] = "{impl1_tag}"
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

### Pattern 3: Smart Constructors

**Problem**: Need to instantiate the correct implementation class from configuration.

**Solution**: Config classes carry metadata about their implementation and provide a factory method.

**Structure**:
```python
class Base{Component}Config(BaseModel):
    # Class-level metadata (not instance data)
    _impl_class: ClassVar[type[Base{Component}]]

    def create_{component}(self) -> Base{Component}:
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

## Implementation Guide

### Step 1: Define Your Domain Model

Identify your polymorphic components:

```
Project Type: {YOUR_PROJECT_TYPE}

Polymorphic Components:
1. {COMPONENT_TYPE_1} (e.g., "database backends", "auth providers")
   Implementations: {IMPL_1A}, {IMPL_1B}, {IMPL_1C}

2. {COMPONENT_TYPE_2} (e.g., "request handlers", "data processors")
   Implementations: {IMPL_2A}, {IMPL_2B}

3. {COMPONENT_TYPE_3} (e.g., "output formatters", "serializers")
   Implementations: {IMPL_3A}, {IMPL_3B}, {IMPL_3C}
```

### Step 2: Create Directory Structure

```
{project_root}/
├── pyproject.toml                    # Poetry config
├── CLAUDE.md                         # Your customized guide
├── src/
│   └── {package_name}/
│       ├── __main__.py              # Application entry point
│       ├── config.py                # Root config class
│       │
│       ├── conf/                    # Hydra configuration files
│       │   ├── config.yaml         # Main config with defaults
│       │   ├── {component1}/       # Config group 1
│       │   │   ├── {impl1a}.yaml
│       │   │   ├── {impl1b}.yaml
│       │   │   └── {impl1c}.yaml
│       │   ├── {component2}/       # Config group 2
│       │   │   ├── {impl2a}.yaml
│       │   │   └── {impl2b}.yaml
│       │   └── hydra/              # Hydra settings
│       │       ├── hydra_logging/
│       │       └── job_logging/
│       │
│       ├── {component1}/           # Component 1 implementations
│       │   ├── __init__.py        # Exports discriminated union
│       │   ├── base_{component1}.py
│       │   ├── base_{component1}_config.py
│       │   ├── {impl1a}.py
│       │   ├── {impl1b}.py
│       │   └── {impl1c}.py
│       │
│       └── {component2}/           # Component 2 implementations
│           ├── __init__.py
│           ├── base_{component2}.py
│           ├── base_{component2}_config.py
│           ├── {impl2a}.py
│           └── {impl2b}.py
└── tests/
```

### Step 3: Set Up Poetry Project

Edit `pyproject.toml`:

```toml
[tool.poetry]
name = "{project-name}"
version = "0.1.0"

[tool.poetry.dependencies]
python = ">=3.11"
pydantic = ">=2.0.0"
hydra-core = ">=1.3.0"
omegaconf = ">=2.3.0"

# Optional dependencies for different implementations
{optional_package_1} = {version = ">=1.0.0", optional = true}
{optional_package_2} = {version = ">=1.0.0", optional = true}

[tool.poetry.extras]
{impl1a} = ["{optional_package_1}"]
{impl1b} = ["{optional_package_2}"]
all = ["{optional_package_1}", "{optional_package_2}"]

[tool.poetry.scripts]
{cli_command} = "{package_name}.__main__:main"
```

### Step 4: Implement Base Classes

**Base Configuration** (`base_{component}_config.py`):

```python
from typing import TYPE_CHECKING, ClassVar
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .base_{component} import Base{Component}

class Base{Component}Config(BaseModel):
    """Base configuration for all {component} implementations."""

    model_config = {"arbitrary_types_allowed": True}

    # Discriminator field (filled by subclasses)
    {discriminator_field}: str = Field(
        description="Type identifier for {component}"
    )

    # Common fields shared by all implementations
    {common_field_1}: {Type1} = Field(description="...")
    {common_field_2}: {Type2} = Field(description="...")

    # Each subclass must define this
    _impl_class: ClassVar[type["Base{Component}"]]

    def create_{component}(self) -> "Base{Component}":
        """Smart constructor: instantiate implementation from config."""
        return self._impl_class(config=self)
```

**Base Implementation** (`base_{component}.py`):

```python
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from .base_{component}_config import Base{Component}Config

class Base{Component}(ABC):
    """Abstract base class for {component} implementations."""

    def __init__(self, config: "Base{Component}Config"):
        """Initialize with configuration."""
        self.config = config

    # Your abstract methods here
    @abstractmethod
    async def {main_method}(self, {params}) -> {ReturnType}:
        """Main functionality of {component}."""
        pass

    # Context manager support (optional but recommended)
    async def __aenter__(self) -> Self:
        return self

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup resources."""
        pass
```

### Step 5: Implement Concrete Classes

**Concrete Config**:

```python
from typing import Literal, ClassVar
from pydantic import Field
from .base_{component}_config import Base{Component}Config

class {Impl}Config(Base{Component}Config):
    """Configuration for {impl} implementation."""

    _impl_class: ClassVar[type[Base{Component}]] = {Impl}

    # Discriminator with Literal type
    {discriminator_field}: Literal["{impl_tag}"] = "{impl_tag}"

    # Implementation-specific fields
    {specific_param_1}: {Type1} = Field(description="...")
    {specific_param_2}: {Type2} = Field(description="...")
```

**Concrete Implementation**:

```python
from .base_{component} import Base{Component}

class {Impl}(Base{Component}):
    """Implementation of {component} using {technology}."""

    def __init__(self, config: "{Impl}Config"):
        super().__init__(config)
        # Initialize resources using config
        self.{resource} = self._initialize_from_config()

    def _initialize_from_config(self):
        """Initialize resources based on configuration."""
        # Access config via self.config
        return SomeResource(self.config.{specific_param_1})

    async def {main_method}(self, {params}) -> {ReturnType}:
        """Implementation of main functionality."""
        # ... implementation ...
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup resources."""
        # ... cleanup ...
        pass
```

### Step 6: Create Discriminated Union

**In `{component}/__init__.py`**:

```python
from typing import Annotated
from pydantic import Field

from .{impl1} import {Impl1}Config
from .{impl2} import {Impl2}Config
from .{impl3} import {Impl3}Config

# Discriminated union type
{Component}Config = Annotated[
    {Impl1}Config
    | {Impl2}Config
    | {Impl3}Config,
    Field(discriminator="{discriminator_field}"),
]
```

### Step 7: Create Root Config

**In `config.py`**:

```python
from pydantic import BaseModel, Field
from {package_name}.{component1} import {Component1}Config
from {package_name}.{component2} import {Component2}Config

class MainConfig(BaseModel):
    """Root configuration for the application."""

    {component1}: {Component1}Config = Field(
        description="{Component1} configuration"
    )
    {component2}: {Component2}Config = Field(
        description="{Component2} configuration"
    )
    # ... other top-level config
```

### Step 8: Create Hydra YAML Files

**Main config** (`conf/config.yaml`):

```yaml
defaults:
  - {component1}: {default_impl1}
  - {component2}: {default_impl2}
  - _self_

hydra:
  run:
    dir: outputs/${hydra.job.override_dirname}
  job:
    chdir: true
```

**Component config** (`conf/{component1}/{impl}.yaml`):

```yaml
# @package {component1}
{discriminator_field}: "{impl_tag}"
{common_field_1}: {value1}
{common_field_2}: {value2}
{specific_param_1}: {value3}
{specific_param_2}: {value4}
```

### Step 9: Create Application Entry Point

**In `__main__.py`**:

```python
import asyncio
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from {package_name}.config import MainConfig

async def async_main(config: MainConfig) -> None:
    """Main application logic."""
    # Get Hydra output directory
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)

    # Create instances using smart constructors
    {component1} = config.{component1}.create_{component1}()
    {component2} = config.{component2}.create_{component2}()

    # Use components
    async with {component1}, {component2}:
        result = await {component1}.{main_method}(...)
        # ... application logic ...

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    # Convert Hydra DictConfig to Pydantic model
    config = MainConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))

    # Run async main
    asyncio.run(async_main(config))

if __name__ == "__main__":
    main()
```

---

## Boilerplate Templates

### Quick Reference: Files to Create

For each new polymorphic component:

```
✓ {component}/base_{component}.py            - Abstract implementation
✓ {component}/base_{component}_config.py     - Abstract config
✓ {component}/{impl}.py                      - Concrete implementation
✓ {component}/__init__.py                    - Discriminated union
✓ conf/{component}/{impl}.yaml               - Hydra defaults
```

### Template: Minimal Implementation

**Config** (~20 lines):
```python
from typing import Literal, ClassVar
from pydantic import Field
from .base_{component}_config import Base{Component}Config

class {Impl}Config(Base{Component}Config):
    _impl_class: ClassVar = {Impl}

    {discriminator}: Literal["{tag}"] = "{tag}"
    param1: {Type} = Field(description="...")
    param2: {Type} = Field(description="...")
```

**Implementation** (~20 lines):
```python
from .base_{component} import Base{Component}

class {Impl}(Base{Component}):
    def __init__(self, config: "{Impl}Config"):
        super().__init__(config)
        self.{resource} = initialize(config.param1, config.param2)

    async def {method}(self, {params}) -> {Return}:
        # TODO: implement
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # Cleanup
        pass
```

**YAML** (~5 lines):
```yaml
# @package {component}
{discriminator}: "{tag}"
param1: {value1}
param2: {value2}
```

---

## Common Variations

### Variation 1: Nested Discriminated Unions

Nest discriminated unions for multi-level polymorphism:

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

### Variation 2: Shared Base Configuration

Extract common config to a shared base:

```python
class SharedConfig(BaseModel):
    """Config shared by multiple components."""
    timeout: float
    retries: int

class MainConfig(BaseModel):
    shared: SharedConfig
    {component1}: {Component1}Config
    {component2}: {Component2}Config
```

Then reference in implementations:
```python
def __init__(self, config: Config, shared: SharedConfig):
    self.timeout = shared.timeout
```

### Variation 3: Synchronous Application

For non-async applications:

```python
def main(cfg: DictConfig) -> None:
    config = MainConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))

    component = config.component.create_component()
    with component:  # Use sync context manager
        result = component.process(...)  # Sync method
```

### Variation 4: No Smart Constructors

If you prefer explicit factory functions:

```python
# Instead of config.create_component()
def create_component(config: ComponentConfig) -> BaseComponent:
    """Factory function."""
    if isinstance(config, Impl1Config):
        return Impl1(config)
    elif isinstance(config, Impl2Config):
        return Impl2(config)
    # ... etc
```

(But smart constructors are generally cleaner!)

### Variation 5: Add Lifecycle Hooks

Extend base classes with lifecycle hooks:

```python
class Base{Component}(ABC):
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

### Variation 6: Custom Validation

Add cross-field validation with Pydantic validators:

```python
from pydantic import model_validator

class {Component}Config(Base{Component}Config):

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        """Validate that fields are consistent."""
        if self.param1 > self.param2:
            raise ValueError("param1 must be <= param2")
        return self
```

---

## Example: This Codebase

This template is derived from the inference benchmarking codebase. Here's how the patterns map:

| Pattern | This Codebase Example |
|---------|----------------------|
| **Polymorphic Component 1** | Backend (MockBackend, TransformersBackend, VLLMBackend, etc.) |
| **Polymorphic Component 2** | Runner (StandardRunner, OptimizationRunner) |
| **Discriminator Field** | `backend_type`, `runner_type` |
| **Smart Constructor** | `create_backend()`, `create_runner()` |
| **Main Config** | `MainConfig` with `backend` and `runner` fields |
| **Entry Point** | `__main__.py:225` with `@hydra.main` |
| **Config Directory** | `src/inference_benchmarks/conf/` |

See the concrete `CLAUDE.md` file for the full detailed example.

---

## Hydra Config Directory Organization

This section describes the organizational pattern for Hydra configuration files. The key principle is **minimal root config with compositional groups**.

### Directory Structure

```
conf/
├── config.yaml              # Minimal root - only mandatory settings
├── hydra/                   # Hydra-specific settings
│   ├── base.yaml           # Standard hydra config (logging, output dirs)
│   ├── job_logging/        # Job logging variants
│   └── hydra_logging/      # Hydra logging variants
│
├── command/                 # Command configs (@package _global_)
│   ├── evaluate.yaml       # command: evaluate + command-specific options
│   ├── optimize.yaml       # command: optimize + command-specific options
│   └── precompute_traces.yaml
│
├── evaluator/              # Evaluator configs (@package evaluator)
│   ├── generative.yaml     # Uses interpolation: ${backend}, ${model.num_layers}
│   ├── importance_weighted.yaml
│   └── mock_supernet.yaml
│
├── backend/                # Backend configs (@package backend)
│   ├── transformers.yaml   # Local HuggingFace model
│   └── vllm.yaml          # Remote vLLM server
│
├── model/                  # Model configs (@package model)
│   ├── 6000.yaml          # /workspace/6000 checkpoint
│   └── qwen2_0_5b.yaml    # Qwen2-0.5B
│
├── task/                   # Task configs (@package task)
│   ├── gsm8k.yaml         # Full GSM8K evaluation
│   └── gsm8k_quick.yaml   # Quick 100-problem subset
│
├── generation/             # Generation configs (@package generation)
│   ├── greedy.yaml        # Deterministic decoding (temperature=0)
│   ├── greedy_short.yaml  # Short context (512 tokens)
│   └── sampling.yaml      # Stochastic sampling
│
├── placement/              # Placement configs (@package _global_)
│   ├── all_attention.yaml # Baseline: all attention layers
│   ├── every_2nd_gdn.yaml # GDN at every 2nd layer
│   ├── every_3rd_gdn.yaml # GDN at every 3rd layer
│   └── every_4th_gdn.yaml # GDN at every 4th layer
│
├── experiments/            # NOT a config group - standalone root configs
│   ├── eval_6000_gsm8k.yaml
│   └── precompute_gsm8k_vllm.yaml
│
└── [problem/, search_strategy/, result_handler/]  # Other config groups
```

### Key Principles

#### 1. Minimal Root Config

The root `config.yaml` should only specify mandatory settings:

```yaml
# config.yaml
defaults:
  - command: ???  # Must specify - forces user to choose a command
  - hydra: base   # Pull in standard hydra settings
  - _self_
```

**Why**: Most config groups will be overridden anyway. Don't set defaults that nobody uses.

#### 2. Command Configs are Global Scope

Command configs use `@package _global_` because different commands have different schemas:

```yaml
# command/evaluate.yaml
# @package _global_

command: evaluate

# Command-specific options with interpolation
mixer_names: ${model.mixer_names}

# Single placement (use Hydra --multirun for sweeps)
# Use placement config group or set to null for non-supernets
placement: null
```

Commands define their structure and can use interpolation to reference other config groups.

#### 3. Component Configs are Scoped

Regular config groups use their package scope:

```yaml
# backend/transformers.yaml
# @package backend

backend_type: transformers
model_path: ${oc.env:MODEL_PATH,/workspace/6000}
torch_dtype: bfloat16
# ...
```

```yaml
# model/6000.yaml
# @package model

model_path: /workspace/6000
num_layers: 24
mixer_names: [attention, gdn, kda, sliding_window]
```

#### 4. Evaluators Use Interpolation for Composition

Evaluator configs wire together other config groups via interpolation:

```yaml
# evaluator/generative.yaml
# @package evaluator

evaluator_type: generative

# Composed from other config groups
backend: ${backend}
num_layers: ${model.num_layers}
mixer_names: ${model.mixer_names}
tasks: ${task.tasks}
num_fewshot: ${task.num_fewshot}
max_problems_per_task: ${task.max_problems_per_task}
split: ${task.split}
samples_per_problem: ${generation.samples_per_problem}
max_new_tokens: ${generation.max_new_tokens}
temperature: ${generation.temperature}
top_p: ${generation.top_p}

# Evaluator-specific (not composed)
scoring_method: any_correct
```

#### 5. Experiments ARE a Config Group (with Global Scope)

The `experiments/` folder is a config group, but experiment configs use `# @package _global_` to inject their contents at root scope.

```yaml
# experiments/eval_6000_gsm8k.yaml
# @package _global_

defaults:
  - /command: evaluate
  - /evaluator: generative
  - /backend: transformers
  - /model: "6000"      # Quoted because it looks like a number
  - /task: gsm8k
  - /generation: greedy_short
  - /placement: all_attention  # Use placement config group
  - _self_

# Experiment-specific overrides
task:
  max_problems_per_task: null  # Use full test set
```

**Key points**:
- Uses `# @package _global_` to put contents at root scope
- Uses absolute paths with `/` prefix: `/command: evaluate` not `command: evaluate`
- Does NOT include `/hydra: base` - inherited from config.yaml
- Pulls in ALL required config groups for the command
- Can override specific values after composition
- Run with: `python -m place_layers +experiments=eval_6000_gsm8k`

### Usage Patterns

#### Pattern A: Run an Experiment

```bash
# Run a pre-configured experiment (recommended)
python -m place_layers +experiments=eval_6000_gsm8k
```

#### Pattern B: Compose on CLI

```bash
# Compose config groups on the command line
python -m place_layers \
    command=evaluate \
    evaluator=generative \
    backend=transformers \
    model=6000 \
    task=gsm8k \
    generation=greedy
```

#### Pattern C: Override Specific Values

```bash
# Use experiment but override specific values
python -m place_layers \
    +experiments=eval_6000_gsm8k \
    task.max_problems_per_task=50
```

#### Pattern D: Sweep Over Placements

```bash
# Evaluate multiple placements using Hydra multirun
python -m place_layers \
    command=evaluate \
    evaluator=generative \
    backend=transformers \
    model=10000 \
    task=gsm8k \
    generation=greedy \
    placement=all_attention,every_2nd_gdn,every_4th_gdn \
    --multirun
```

### Hydra Base Config

Centralize all Hydra settings in `hydra/base.yaml`:

```yaml
# hydra/base.yaml
# @package hydra

defaults:
  - override job_logging: rich_files
  - override hydra_logging: rich

run:
  dir: outputs/${hydra.job.override_dirname}

sweep:
  dir: multirun/${hydra.job.override_dirname}
  subdir: ${hydra.job.num}

job:
  chdir: true
  config:
    override_dirname:
      exclude_keys:
        - hydra.run.dir
        - hydra.sweep.dir
        - hydra.output_subdir
        - hydra.job.chdir
        - output_path
```

### Common Gotchas

**Numeric config names must be quoted**:
```yaml
defaults:
  - /model: "6000"  # ✓ Quoted - treated as string
  - /model: 6000    # ✗ Unquoted - parsed as integer, fails
```

**Use `override` for hydra sub-defaults**:
```yaml
# In hydra/base.yaml
defaults:
  - override job_logging: rich_files   # ✓ Override existing
  - job_logging: rich_files            # ✗ Conflicts with built-in
```

**Absolute paths in experiments**:
```yaml
# In experiments/*.yaml
defaults:
  - /command: evaluate    # ✓ Absolute path from conf root
  - command: evaluate     # ✗ Relative - looks in experiments/command/
```

---

## Troubleshooting

### Common Issues

**Problem**: `ValidationError: Discriminator 'X' is missing`
- **Cause**: YAML file doesn't include discriminator field
- **Fix**: Add `{discriminator_field}: "{value}"` to YAML

**Problem**: `TypeError: missing 1 required positional argument: 'config'`
- **Cause**: Trying to instantiate implementation directly without config
- **Fix**: Use smart constructor: `config.create_{component}()`

**Problem**: Circular import between config and implementation
- **Cause**: Both files import each other
- **Fix**: Use `TYPE_CHECKING` guard and string type annotations

**Problem**: `KeyError` when accessing nested config
- **Cause**: Hydra defaults not properly composed
- **Fix**: Check `defaults:` list and `@package` annotations

**Problem**: Hydra creating wrong output directory structure
- **Cause**: `hydra.run.dir` not configured properly
- **Fix**: Set in `conf/config.yaml` under `hydra.run.dir`

---

## Checklist: Adding a New Implementation

Use this checklist when adding a new implementation to an existing component:

- [ ] Create config class `{Impl}Config` with:
  - [ ] `_impl_class` ClassVar
  - [ ] Discriminator with Literal type
  - [ ] Implementation-specific fields
- [ ] Create implementation class `{Impl}`:
  - [ ] `__init__(self, config)` constructor
  - [ ] Override all abstract methods
  - [ ] `__aexit__` cleanup (if using context manager)
- [ ] Add to discriminated union in `__init__.py`
- [ ] Create Hydra YAML in `conf/{component}/{impl}.yaml`
- [ ] Update `pyproject.toml` if new dependencies:
  - [ ] Add to `[tool.poetry.dependencies]` with `optional = true`
  - [ ] Add to `[tool.poetry.extras]`
- [ ] Test:
  - [ ] Config validates without errors
  - [ ] Smart constructor creates correct instance
  - [ ] Hydra override works: `{component}={impl}`

---

## Core Principles for Stateful Applications

### Principle 1: State is Always a Pydantic BaseModel

**All stateful objects must be Pydantic BaseModels.**

Any object that needs to be serialized, checkpointed, or transmitted must inherit from `pydantic.BaseModel`. This is a hard requirement for persistence and recovery.

**Why**: Pydantic provides built-in serialization/deserialization that handles complex nested structures, type validation, and forward references. This ensures that state can be reliably saved and restored.

**Examples**:

✅ **Correct**:
```python
from pydantic import BaseModel

class State(BaseModel):
    """State that can be serialized."""
    model_config = {"frozen": True}  # Immutable if needed
    field1: str
    field2: int
    nested: dict[str, list[int]]
```

❌ **Incorrect**:
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class State:  # ❌ Can't checkpoint reliably
    field1: str
    field2: int
```

### Principle 2: Always Use Pydantic's Idiomatic Serialization

**Never write custom serialization. Always use Pydantic's built-in methods.**

Use `model_dump()`, `model_dump_json()`, and `model_validate()` for all serialization needs.

**Why**: Custom serialization is error-prone and breaks forward compatibility. Pydantic's serialization handles nested models, custom types (Path, datetime), validation, and edge cases.

**Examples**:

✅ **Correct**:
```python
from pydantic import BaseModel
import json

class Config(BaseModel):
    param1: str
    param2: int

# Serialization
config = Config(param1="value", param2=42)
json_str = config.model_dump_json(indent=2)
dict_data = config.model_dump()

# Deserialization
loaded = Config.model_validate_json(json_str)
# or
loaded = Config.model_validate(dict_data)

# Saving to file
with open("config.json", "w") as f:
    f.write(config.model_dump_json(indent=2))

# Loading from file
with open("config.json", "r") as f:
    loaded = Config.model_validate_json(f.read())
```

❌ **Incorrect**:
```python
# ❌ Don't manually construct dictionaries for serialization
def to_dict(config):
    return {
        "param1": config.param1,
        "param2": config.param2,
    }

# ❌ Don't bypass validation
config = Config(**raw_dict)  # Might work but skips validators!

# ❌ Don't write custom JSON encoders
class ConfigEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Config):
            return {"param1": obj.param1, "param2": obj.param2}
        return super().default(obj)

# ❌ Don't use custom Pydantic serializers unless absolutely necessary
from pydantic import model_serializer

class Config(BaseModel):
    data: tuple[int, ...]

    @model_serializer  # ❌ Unnecessary! Pydantic handles tuples automatically
    def serialize_model(self):
        return {"data": list(self.data)}
```

---

## Anti-Patterns to Avoid

❌ **Don't add defaults to Pydantic classes**
```python
class Config(BaseModel):
    param: int = 42  # ❌ NO! Put in YAML instead
```

✅ **Do put defaults in YAML**
```yaml
param: 42  # ✓ YES!
```

---

❌ **Don't use string discriminators without Literal**
```python
backend_type: str = "mock"  # ❌ Type checker can't verify
```

✅ **Do use Literal types**
```python
backend_type: Literal["mock"] = "mock"  # ✓ Type safe!
```

---

❌ **Don't mix config and implementation in one file**
```python
# same_file.py
class Config(BaseModel): ...
class Implementation: ...  # ❌ Hard to test, circular imports
```

✅ **Do separate config and implementation**
```python
# config.py
class Config(BaseModel): ...

# implementation.py
class Implementation: ...  # ✓ Clean separation
```

---

❌ **Don't skip validation**
```python
def create_thing(cfg: DictConfig):
    return Thing(cfg["param"])  # ❌ No validation!
```

✅ **Do validate with Pydantic**
```python
config = ThingConfig.model_validate(OmegaConf.to_container(cfg))
return config.create_thing()  # ✓ Validated!
```

---

❌ **Don't maintain `__all__` in `__init__.py` files**
```python
# __init__.py
from .module import Thing1, Thing2, Thing3

__all__ = ["Thing1", "Thing2", "Thing3"]  # ❌ Extra maintenance, not necessary
```

✅ **Do let Python handle exports naturally**
```python
# __init__.py
from .module import Thing1, Thing2, Thing3

# That's it! No __all__ needed. Modern IDEs and type checkers handle this fine.
```

**Why**: `__all__` is extra maintenance burden with minimal benefit in modern Python with type hints. Let imports be explicit at the use site.

---

## Summary

This template provides a battle-tested architecture for building configurable, type-safe Python applications. The three core patterns work together to create a robust system:

1. **Configuration Separation** - Externalize all defaults to YAML
2. **Discriminated Unions** - Type-safe polymorphism
3. **Smart Constructors** - Config-driven instantiation

Use this template as a starting point for your own projects. Customize the placeholders, adapt to your domain, and enjoy the benefits of a well-structured configuration system!

---

## Fast-LLM ForwardKL Evaluator Configuration

This section documents how to configure the ForwardKL evaluator in Fast-LLM for importance-weighted evaluation during training.

### Overview

The ForwardKL evaluator computes the forward KL divergence between a teacher model's log-probabilities (pre-computed traces) and the student model being trained. This is used for importance-weighted accuracy estimation.

### Prerequisites

1. **Pre-computed traces**: Generate traces using the `precompute_traces` command:
   ```bash
   # Start vLLM server with teacher model
   python -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen2-0.5B \
       --port 8000 \
       --max-model-len 4096 \
       --gpu-memory-utilization 0.8

   # Generate traces
   VLLM_MODEL="Qwen/Qwen2-0.5B" python -m place_layers \
       command=precompute_traces \
       "tasks=[gsm8k]" \
       max_problems_per_task=null \
       samples_per_problem=8 \
       max_new_tokens=1024
   ```

2. **Output structure**: The traces are saved in HuggingFace-compatible format:
   ```
   output_dir/
       validation.jsonl    # All valid traces (load with datasets)
       skipped.jsonl       # Skipped problems
       metadata.json       # Provenance info
   ```

3. **Rename output folder**: Remove special characters (brackets) from the Hydra output folder name to ensure HuggingFace compatibility:
   ```bash
   mv "outputs/command=precompute_traces,..." "outputs/gsm8k_traces"
   ```

### YAML Configuration

Add a `forward_kl` evaluator to your Fast-LLM training config:

```yaml
training:
  train_iters: 1000
  logs:
    interval: 10
  evaluators:
    # Standard loss evaluation (optional)
    validation:
      evaluator:
        type: loss
        iterations: 100

    # Forward KL evaluation against teacher traces
    forward_kl_gsm8k:
      evaluator:
        type: forward_kl
        dataset_path: "/path/to/gsm8k_traces"  # Directory with validation.jsonl
        task: null                              # Optional: dataset config name
        num_samples: null                       # null = use all traces
        batch_size: 8                           # Batch size for forward passes
        trust_remote_code: false                # For custom datasets

batch:
  sequence_length: 4096        # Must be >= max trace length
  micro_batch_size: 2
  batch_size: 64

# ... rest of config (model, optimizer, etc.)
```

### Config Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | str | - | Must be `"forward_kl"` |
| `dataset_path` | str | null | Path to directory containing `validation.jsonl` |
| `task` | str | null | HuggingFace dataset config/task name (optional) |
| `num_samples` | int | null | Limit number of traces (null = all) |
| `batch_size` | int | 8 | Batch size for forward passes |
| `trust_remote_code` | bool | false | Trust remote code when loading dataset |

### Distributed Training Support

The ForwardKL evaluator supports all Fast-LLM parallelism modes:

- **Data Parallel (DP)**: Traces are sharded across DP ranks using HuggingFace's `.shard()`. Results are reduced across the data group.
- **Tensor Parallel (TP)**: Works transparently - all TP ranks process the same traces.
- **Sequence Parallel (SP)**: Traces are padded to training `sequence_length` and processed with proper SP splitting.
- **Pipeline Parallel (PP)**: Only the last PP stage computes logits. Results are reduced across the pipeline group.

### Output Metrics

The evaluator logs:
```
Forward KL (forward_kl_gsm8k): 2.3456 (3708 traces) [57 skipped]
```

Metrics are also logged to wandb/tensorboard under:
- `validation.forward_kl_gsm8k.forward_kl`
- `validation.forward_kl_gsm8k.num_traces`
- `validation.forward_kl_gsm8k.iteration`

### Trace Format

Each trace in `validation.jsonl` must have:
```json
{
  "record_type": "trace",
  "prompt_tokens": [1, 2, 3, ...],
  "completion_tokens": [4, 5, 6, ...],
  "correct": true,
  "teacher_log_prob": -12.345,
  "problem_id": "gsm8k:42",
  "task_name": "gsm8k",
  "sample_idx": 0,
  "completion_text": "The answer is 42."
}
```

### Example: Full Training Config

```yaml
training:
  train_iters: 10000
  num_workers: 8
  logs:
    interval: 100
  evaluators:
    forward_kl:
      evaluator:
        type: forward_kl
        dataset_path: "/workspace/outputs/gsm8k_traces"
        batch_size: 16

batch:
  sequence_length: 4096
  micro_batch_size: 4
  batch_size: 128

data:
  datasets:
    training:
      type: file
      path: /path/to/training/data

optimizer:
  learning_rate:
    base: 1.0e-05
    decay_style: cosine
    warmup_iterations: 100
  weight_decay: 0.1
  beta_1: 0.9
  beta_2: 0.95

model:
  base_model:
    # ... model architecture config
  multi_stage:
    zero_stage: 2
  distributed:
    compute_dtype: bf16
    tensor_parallel: 1
    pipeline_parallel: 1
    data_parallel: 8
    sequence_parallel: false

run:
  experiment_dir: training_with_forward_kl
```
