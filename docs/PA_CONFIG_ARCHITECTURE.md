# PA Configuration Architecture

This document explains how PA orchestrator configuration is structured and managed.

## Overview

The PA orchestrator configuration is split into two independent files:

```
config/pa_defaults.yaml      → PA-specific settings (version, preemption, etc.)
     +
user_job_spec.yaml          → User-specific settings (resources, mounts, env)
     ↓
submit_pa_orchestrator.py   → Merges both configs
     ↓
merged_spec.yaml            → Final job specification submitted to cluster
```

## Configuration Files

### 1. PA Defaults (`config/pa_defaults.yaml`)

**Purpose**: PA API-specific settings that rarely change

**Managed by**: Project maintainers

**Update frequency**: Only when PA API changes

**Contents**:
```yaml
isProcessAgent: true
options:
  sidecars:
    process-agent:
      version: 3  # PA API version
preemptable: false
```

**Location**: `config/pa_defaults.yaml`

**Version Control**: Checked into git

### 2. User Job Spec

**Purpose**: User/environment-specific settings

**Managed by**: End users

**Update frequency**: Per project/environment

**Contents**:
```yaml
cpu: 4                    # PA orchestrator resources
mem: 16
image: ubuntu
data:                     # Data mounts
  - servicenow.data.home:/home/toolkit
env:                      # Environment variables
  PYTHONPATH: /path
  LOG_LEVEL: INFO
```

**Location**: User-specified (passed via `--job-spec`)

**Version Control**: Not checked in (user-specific)

## Merge Process

The `submit_pa_orchestrator.py` script merges these configs:

```python
# 1. Load PA defaults
pa_config = load_yaml("config/pa_defaults.yaml")

# 2. Load user spec
user_spec = load_yaml(args.job_spec)

# 3. Transform user spec
# - env dict → environmentVars array
# - workingDir → workdir
# - cpu/mem/gpu → resources object

# 4. Merge (PA defaults override user values)
merged = user_spec | pa_config

# 5. Add orchestrator command
merged["command"] = ["bash", "-c", "python -m place_layers ..."]

# 6. Write final spec
write_yaml(merged, args.output_spec)
```

## Field Transformations

The script automatically transforms user-friendly fields to PA API format:

| User Input | PA API | Transformation |
|------------|--------|----------------|
| `env: {KEY: val}` | `environmentVars: ["KEY=val"]` | Dict to array |
| `workingDir: /path` | `workdir: /path` | Rename |
| `cpu: 4, mem: 16` | `resources: {cpu: 4, mem: 16}` | Wrap in object |

## Usage Examples

### Basic Usage (Default PA Config)

```bash
python scripts/submit_pa_orchestrator.py \
    --job-spec my_spec.yaml \
    --model-name apriel2_0_5b_dev \
    --num-children 4
```

Uses `config/pa_defaults.yaml` automatically.

### Custom PA Config

```bash
# Create custom PA config with different version
cat > custom_pa.yaml <<EOF
isProcessAgent: true
options:
  sidecars:
    process-agent:
      version: 4  # Newer PA version
preemptable: true  # Allow preemption
EOF

# Use custom config
python scripts/submit_pa_orchestrator.py \
    --job-spec my_spec.yaml \
    --pa-config custom_pa.yaml \
    --model-name apriel2_0_5b_dev \
    --num-children 4
```

## Updating PA Configuration

### Scenario 1: PA API Version Update

When PA API version changes from v3 to v4:

**Steps:**
1. Edit `config/pa_defaults.yaml`:
   ```yaml
   options:
     sidecars:
       process-agent:
         version: 4  # Updated from 3
   ```
2. Commit and push
3. All users automatically get the new version

**No code changes needed!**

### Scenario 2: Add New PA Option

When PA adds a new required option (e.g., `maxRetries`):

**Steps:**
1. Edit `config/pa_defaults.yaml`:
   ```yaml
   options:
     sidecars:
       process-agent:
         version: 3
         maxRetries: 3  # New option
   ```
2. Commit and push
3. All users automatically get the new option

**No code changes needed!**

### Scenario 3: Per-User Override

When a specific user needs different settings:

**Steps:**
1. User creates custom PA config:
   ```yaml
   # my_custom_pa.yaml
   isProcessAgent: true
   options:
     sidecars:
       process-agent:
         version: 3
   preemptable: true  # Different from default
   ```
2. User runs with `--pa-config`:
   ```bash
   python scripts/submit_pa_orchestrator.py \
       --job-spec my_spec.yaml \
       --pa-config my_custom_pa.yaml \
       ...
   ```

**No changes to repo needed!**

## Benefits

### 1. Separation of Concerns

- PA settings: Managed by maintainers
- User settings: Managed by users
- No conflicts or confusion

### 2. Easy Updates

- PA version updates: Edit one YAML file
- No Python code changes required
- No redeployment needed

### 3. Override Flexibility

- Default config for 99% of users
- Custom config for special cases
- No forking or patching needed

### 4. Version Control

- PA defaults tracked in git
- User specs not tracked (personal/environment-specific)
- Clear history of PA configuration changes

## Troubleshooting

### PA Config Not Found

**Error:**
```
FileNotFoundError: config/pa_defaults.yaml not found
```

**Solution:**
```bash
# Check file exists
ls config/pa_defaults.yaml

# Or specify custom location
python scripts/submit_pa_orchestrator.py \
    --pa-config /path/to/pa_defaults.yaml \
    ...
```

### Invalid PA Config

**Error:**
```
KeyError: 'isProcessAgent'
```

**Solution:**
Ensure PA config has required fields:
```yaml
isProcessAgent: true
options:
  sidecars:
    process-agent:
      version: 3
preemptable: false
```

### Merge Conflict

**Problem:** PA config overwrites user settings

**Solution:** PA config is applied last, so it intentionally overrides user settings. This ensures PA requirements are always met. If you need different PA settings, use `--pa-config` with a custom file.

## Future Enhancements

Potential improvements:

1. **Schema Validation**: Validate PA config against OpenAPI schema
2. **Version Detection**: Auto-detect PA version from cluster
3. **Config Templates**: Provide templates for common scenarios
4. **Merge Strategies**: Allow fine-grained control over merge behavior
