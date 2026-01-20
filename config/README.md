# Configuration Files

This directory contains default configuration files used by the project.

## pa_defaults.yaml

Default Process Agent configuration used by `scripts/submit_pa_orchestrator.py`.

### Contents

```yaml
isProcessAgent: true
options:
  sidecars:
    process-agent:
      version: 3  # PA API version
preemptable: false
```

### When to Modify

Update this file when:
- **PA API version changes**: Update `options.sidecars.process-agent.version`
- **Preemption policy changes**: Modify `preemptable` setting
- **Additional PA options needed**: Add under `options`

### Usage

The helper script uses this file by default:
```bash
python scripts/submit_pa_orchestrator.py --job-spec my_spec.yaml ...
```

To use a custom PA config:
```bash
python scripts/submit_pa_orchestrator.py \
    --job-spec my_spec.yaml \
    --pa-config /path/to/custom_pa_defaults.yaml \
    ...
```

### Version History

| Version | Changes |
|---------|---------|
| 3 | Initial configuration (current) |

## Adding New Config Files

When adding new configuration files to this directory:

1. **Document the purpose**: Add a section to this README
2. **Use clear naming**: `{component}_defaults.yaml`
3. **Include comments**: Explain each field in the YAML file
4. **Version if needed**: Track breaking changes
