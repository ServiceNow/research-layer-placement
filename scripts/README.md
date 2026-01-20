# Scripts

## submit_pa_orchestrator.py

Helper script to submit PA orchestrator jobs to the intercan cluster.

### Configuration

The script uses two configuration files:

1. **PA Defaults** (`config/pa_defaults.yaml`): PA-specific settings (version, preemption, etc.)
   - Override with `--pa-config` if needed
   - Edit this file when PA API version changes

2. **User Job Spec**: Your job requirements (resources, mounts, environment)
   - Passed via `--job-spec` argument
   - Create one per project/environment

### Quick Start

```bash
# Create your job spec
cat > my_job_spec.yaml <<EOF
cpu: 4
mem: 16
image: ubuntu
data:
  - servicenow.data.home:/home/toolkit
env:
  PYTHONPATH: /home/toolkit/code/research-layer-placement/src
EOF

# Submit PA orchestrator (uses default config/pa_defaults.yaml)
python scripts/submit_pa_orchestrator.py \
    --job-spec my_job_spec.yaml \
    --model-name apriel2_0_5b_dev \
    --num-children 4 \
    --child-mem 32 \
    --child-gpu 1
# Note: --child-image defaults to the cluster's interactive-toolkit image

# Or use custom PA config
python scripts/submit_pa_orchestrator.py \
    --job-spec my_job_spec.yaml \
    --pa-config /path/to/custom_pa_config.yaml \
    --model-name apriel2_0_5b_dev \
    --num-children 4
```

### Full Documentation

See [docs/PA_ORCHESTRATOR.md](../docs/PA_ORCHESTRATOR.md) for complete documentation.

### Dependencies

- PyYAML: `pip install pyyaml`
- eai CLI: Must be installed and configured
