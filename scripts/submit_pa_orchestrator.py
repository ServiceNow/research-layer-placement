#!/usr/bin/env python
"""Helper script to submit PA orchestrator jobs.

This script:
1. Merges user job spec with PA-specific configuration
2. Submits the PA orchestrator job via eai
3. Waits for job to start and get an IP
4. Prints the PA URL for use with pa_rest backend

Usage:
    python scripts/submit_pa_orchestrator.py \
        --job-spec my_job_spec.yaml \
        --model-name apriel2_0_5b_dev \
        --num-children 4 \
        --child-mem 32 \
        --child-gpu 1 \
        --child-image ubuntu \
        --health-check-interval 30 \
        --min-spawn-interval 60 \
        --output-spec /tmp/pa_spec.yaml

Example job spec (my_job_spec.yaml):
    cpu: 4
    mem: 16
    data:
      - servicenow.data.home:/home/toolkit
    env:
      PYTHONPATH: /home/toolkit/code
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# Default location for PA configuration
DEFAULT_PA_CONFIG = Path(__file__).parent.parent / "config" / "pa_defaults.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def merge_specs(user_spec: dict[str, Any], pa_config: dict[str, Any]) -> dict[str, Any]:
    """Merge user job spec with PA-specific configuration.

    Args:
        user_spec: User-provided job spec (cpu, mem, data, env, etc.)
        pa_config: PA-specific settings (isProcessAgent, sidecars, etc.)

    Returns:
        Merged spec ready for submission.
    """
    merged = user_spec.copy()

    # Convert env dict to environmentVars array if not already present
    if "env" in merged and "environmentVars" not in merged:
        env_dict = merged.pop("env")
        merged["environmentVars"] = [f"{k}={v}" for k, v in env_dict.items()]
    elif "env" in merged and "environmentVars" in merged:
        # Both present - append env dict to existing environmentVars
        env_dict = merged.pop("env")
        existing = merged.get("environmentVars", [])
        merged["environmentVars"] = existing + [f"{k}={v}" for k, v in env_dict.items()]

    # Convert workingDir to workdir if present and workdir not set
    if "workingDir" in merged and "workdir" not in merged:
        merged["workdir"] = merged.pop("workingDir")
    elif "workingDir" in merged:
        # workdir already set, remove workingDir
        merged.pop("workingDir")

    # Wrap resources if they're top-level AND resources object doesn't exist
    if "resources" not in merged:
        resources = {}
        for key in ["cpu", "mem", "gpu", "gpuMem"]:
            if key in merged:
                resources[key] = merged.pop(key)
        if resources:
            merged["resources"] = resources
    else:
        # Resources object exists - remove any top-level resource fields
        for key in ["cpu", "mem", "gpu", "gpuMem"]:
            if key in merged:
                merged.pop(key)

    # Add PA-specific fields (including preemptable from PA config)
    merged.update(pa_config)

    return merged


def create_pa_spec(
    user_spec_path: Path,
    model_name: str,
    num_children: int,
    child_mem: int,
    child_gpu: int,
    child_image: str,
    child_port_start: int,
    health_check_interval: int,
    min_spawn_interval: int,
    max_retries: int,
    output_path: Path,
    pa_config_path: Path,
) -> Path:
    """Create PA job spec file.

    Args:
        user_spec_path: Path to user-provided job spec.
        model_name: Model config name (e.g., "apriel2_0_5b_dev").
        num_children: Number of child serve jobs.
        child_mem: Memory per child (GB).
        child_gpu: GPUs per child.
        child_image: Container image for children.
        child_port_start: Starting port for children.
        health_check_interval: Health check interval (seconds).
        min_spawn_interval: Minimum seconds between spawn attempts for same child.
        max_retries: Maximum retry attempts for failed/succeeded jobs.
        output_path: Where to write the merged spec.
        pa_config_path: Path to PA defaults config file.

    Returns:
        Path to the created spec file.
    """
    # Load user spec
    user_spec = load_yaml(user_spec_path)

    # Load PA-specific configuration from file
    pa_config = load_yaml(pa_config_path)
    print(f"Loaded PA config from: {pa_config_path}")

    # Build orchestrator command
    orchestrator_cmd = (
        f"python -m place_layers "
        f"+command=pa_orchestrator "
        f"model_name={model_name} "
        f"num_children={num_children} "
        f"child_spec.mem={child_mem} "
        f"child_spec.gpu={child_gpu} "
        f"child_spec.image={child_image} "
        f"child_spec.port_start={child_port_start} "
        f"health_check_interval={health_check_interval} "
        f"min_spawn_interval={min_spawn_interval} "
        f"max_retries={max_retries}"
    )

    # Add command to spec
    pa_config["command"] = ["bash", "-c", orchestrator_cmd]

    # Merge specs
    merged_spec = merge_specs(user_spec, pa_config)

    # Write to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(merged_spec, f, default_flow_style=False)

    print(f"Created PA spec: {output_path}")
    print(f"Orchestrator command: {orchestrator_cmd}")

    return output_path


def submit_job(spec_path: Path) -> str:
    """Submit job via eai and return job ID.

    Args:
        spec_path: Path to job spec file.

    Returns:
        Job ID.

    Raises:
        RuntimeError: If job submission fails.
    """
    print(f"Submitting job with spec: {spec_path}")

    result = subprocess.run(
        ["eai", "job", "new", "--format", "json", "-f", str(spec_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Job submission failed: {result.stderr}")

    # Parse job ID from output
    # Assuming eai outputs something like "Job ID: abc123" or JSON
    output = result.stdout.strip()
    print(f"Job submission output: {output}")

    # Try to parse as JSON first
    try:
        data = json.loads(output)
        if "id" in data:
            return data["id"]
        elif "job_id" in data:
            return data["job_id"]
    except json.JSONDecodeError:
        pass

    # Try to extract from text output
    for line in output.split("\n"):
        if "job" in line.lower() and "id" in line.lower():
            # Extract ID (assuming format like "Job ID: abc123")
            parts = line.split(":")
            if len(parts) >= 2:
                return parts[-1].strip()

    # If we can't find the ID, return the full output
    # The user might need to adjust this based on actual eai output format
    print("Warning: Could not parse job ID from output, returning full output")
    return output


def get_job_info(job_id: str) -> dict[str, Any]:
    """Get job information via eai.

    Args:
        job_id: Job ID.

    Returns:
        Job info dict with 'state' and 'ip' fields extracted from latest run.

    Raises:
        RuntimeError: If query fails.
    """
    result = subprocess.run(
        ["eai", "job", "info", job_id],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to query job: {result.stderr}")

    # Parse YAML output
    try:
        data = yaml.safe_load(result.stdout)

        # Extract top-level state
        state = data.get("state", "unknown")

        # Extract IP from the latest run (last item in runs list)
        ip = None
        runs = data.get("runs", [])
        if runs:
            latest_run = runs[-1]  # Latest run is the last one
            ip = latest_run.get("ip")

        return {"state": state, "ip": ip}
    except yaml.YAMLError as e:
        raise RuntimeError(f"Failed to parse job info YAML: {e}")


def wait_for_job_ip(job_id: str, timeout: int = 300, poll_interval: int = 5) -> str:
    """Wait for job to get an IP address.

    Args:
        job_id: Job ID.
        timeout: Maximum time to wait (seconds).
        poll_interval: Time between status checks (seconds).

    Returns:
        Job IP address.

    Raises:
        TimeoutError: If job doesn't get IP within timeout.
        RuntimeError: If job fails.
    """
    print(f"Waiting for job {job_id} to get IP address...")

    start_time = time.time()
    while time.time() - start_time < timeout:
        info = get_job_info(job_id)
        state = info.get("state", "unknown")
        ip = info.get("ip")

        print(f"Job state: {state}, IP: {ip}")

        # Check if job failed (using PA API JobState enum values)
        if state in ["FAILED", "CANCELLED", "INTERRUPTED"]:
            raise RuntimeError(f"Job {job_id} failed with state: {state}")

        # Check if we got an IP
        if ip and ip != "unknown" and ip != "None" and ip != "":
            return ip

        time.sleep(poll_interval)

    raise TimeoutError(f"Job {job_id} did not get IP within {timeout} seconds")


def main():
    parser = argparse.ArgumentParser(
        description="Submit PA orchestrator job",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--job-spec",
        type=Path,
        required=True,
        help="Path to user job spec (cpu, mem, data, env, etc.)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model config name (e.g., apriel2_0_5b_dev)",
    )

    # Orchestrator configuration
    parser.add_argument(
        "--num-children",
        type=int,
        default=1,
        help="Number of child serve jobs (default: 1)",
    )
    parser.add_argument(
        "--child-mem",
        type=int,
        default=32,
        help="Memory per child in GB (default: 32)",
    )
    parser.add_argument(
        "--child-gpu",
        type=int,
        default=1,
        help="GPUs per child (default: 1)",
    )
    parser.add_argument(
        "--child-image",
        type=str,
        default="registry.toolkit-sp.yul201.service-now.com/snow.research.afm/interactive-toolkit:25.12-py3-vllm014rc1",
        help="Container image for children",
    )
    parser.add_argument(
        "--child-port-start",
        type=int,
        default=8080,
        help="Starting port for children (default: 8080)",
    )
    parser.add_argument(
        "--health-check-interval",
        type=int,
        default=30,
        help="Health check interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--min-spawn-interval",
        type=int,
        default=60,
        help="Minimum seconds between spawn attempts for same child (default: 60)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retry attempts for failed/succeeded jobs (default: 5, 0=unlimited)",
    )

    # Output options
    parser.add_argument(
        "--output-spec",
        type=Path,
        default=Path("/tmp/pa_orchestrator_spec.yaml"),
        help="Where to write merged spec (default: /tmp/pa_orchestrator_spec.yaml)",
    )
    parser.add_argument(
        "--pa-config",
        type=Path,
        default=DEFAULT_PA_CONFIG,
        help=f"Path to PA defaults config file (default: {DEFAULT_PA_CONFIG})",
    )
    parser.add_argument(
        "--pa-port",
        type=int,
        default=8080,
        help="PA orchestrator port (default: 8080)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for job to start",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Max time to wait for job IP (seconds, default: 300)",
    )

    args = parser.parse_args()

    try:
        # Create PA spec
        spec_path = create_pa_spec(
            user_spec_path=args.job_spec,
            model_name=args.model_name,
            num_children=args.num_children,
            child_mem=args.child_mem,
            child_gpu=args.child_gpu,
            child_image=args.child_image,
            child_port_start=args.child_port_start,
            health_check_interval=args.health_check_interval,
            min_spawn_interval=args.min_spawn_interval,
            max_retries=args.max_retries,
            output_path=args.output_spec,
            pa_config_path=args.pa_config,
        )

        # Submit job
        job_id = submit_job(spec_path)
        print(f"\nJob submitted: {job_id}")

        if args.no_wait:
            print("\nRun 'eai job show {job_id}' to check status")
            return

        # Wait for IP
        ip = wait_for_job_ip(job_id, timeout=args.timeout)

        # Print PA URL
        pa_url = f"http://{ip}:{args.pa_port}"
        print(f"\n{'='*60}")
        print(f"PA Orchestrator is ready!")
        print(f"{'='*60}")
        print(f"Job ID: {job_id}")
        print(f"PA URL: {pa_url}")
        print(f"\nUse this backend config:")
        print(f"  +backend=pa_rest backend.pa_url='{pa_url}'")
        print(f"\nExample evaluation command:")
        print(f"  python -m place_layers \\")
        print(f"      +command=evaluate \\")
        print(f"      +backend=pa_rest \\")
        print(f"      backend.pa_url='{pa_url}' \\")
        print(f"      +evaluator=generative \\")
        print(f"      +task=gsm8k \\")
        print(f"      +generation=greedy")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
