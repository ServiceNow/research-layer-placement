"""PA Orchestrator command - manages multiple serve processes via Process Agent.

This command runs as a Process Agent job and spawns child serve jobs,
providing a single endpoint for dynamic server discovery with auto-recovery.

Usage:
    # Submit PA orchestrator job
    eai job new -f pa_spec.yaml

    # PA orchestrator exposes:
    #   GET /servers -> {"servers": ["http://child1_ip:8080", ...]}
    #   GET /health -> {"status": "ok", "active_children": N}
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, ClassVar, Literal

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, Field

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig

if TYPE_CHECKING:
    from place_layers.backend.base import TransformersBackendConfig

logger = logging.getLogger(__name__)

# Process Agent constants
PA_BASE_URL = "http://0.0.0.0:8666/api/v1"

# Child job states that indicate failure/termination (from PA API JobState enum)
# Note: INTERRUPTED and CANCELLING are NOT included because children have
# preemptable=true and restartable=true, so the cluster handles restarts automatically
FAILED_STATES = {"CANCELLED", "FAILED", "SUCCEEDED"}


class PAChildSpec(BaseModel):
    """Specification for a child serve job."""

    cpu: int = Field(default=2, description="Number of CPU cores", ge=1)
    mem: int = Field(default=16, description="Memory in GB", ge=1)
    gpu: int = Field(description="Number of GPUs", ge=1)
    gpu_mem: int | None = Field(default=None, description="Minimum GPU memory in GB (optional)", ge=1)
    image: str = Field(description="Container image to use")
    data: list[str] = Field(
        default_factory=list,
        description="Volume mounts in format: 'source:destination:mode' (e.g., 'snow.home.user:/home/toolkit:rw')",
    )
    port_start: int = Field(
        default=8080, description="Starting port for serve jobs", ge=1, le=65535
    )


class PAOrchestratorCommand(BaseCommand):
    """Command to orchestrate multiple serve processes via Process Agent."""

    config: "PAOrchestratorCommandConfig"

    async def run(self) -> None:
        """Start PA orchestrator server.

        This runs indefinitely, spawning and monitoring child serve jobs.
        """
        import uvicorn

        # Create FastAPI app
        app = self._create_app()

        # Start health monitoring in background
        asyncio.create_task(self._health_monitor_loop())

        # Configure and run server
        config = uvicorn.Config(
            app,
            host=self.config.host,
            port=self.config.port,
            log_level="info",
        )
        server = uvicorn.Server(config)

        logger.info(f"PA Orchestrator starting on {self.config.host}:{self.config.port}")
        logger.info(f"Managing {self.config.num_children} child serve jobs")

        await server.serve()

    def _create_app(self) -> FastAPI:
        """Create FastAPI app with /servers and /health endpoints."""
        app = FastAPI(title="PA Orchestrator")

        # Track state per index (simplified approach)
        # Each index (0 to num_children-1) has exactly one "current" job
        self._index_info: dict[int, dict] = {}
        self._lock = asyncio.Lock()

        @app.on_event("startup")
        async def startup():
            """Spawn initial child jobs on startup."""
            logger.info("Spawning initial child jobs...")
            await self._spawn_all_children()

        @app.get("/servers")
        async def get_servers():
            """Return list of active server URLs."""
            async with self._lock:
                servers = [
                    f"http://{info['ip']}:{info['port']}"
                    for info in self._index_info.values()
                    if info["state"] == "RUNNING" and info["ip"]
                ]
            logger.debug(f"Returning {len(servers)} active servers")
            return {"servers": servers}

        @app.get("/health")
        async def get_health():
            """Return PA orchestrator health status."""
            async with self._lock:
                active = sum(
                    1 for info in self._index_info.values() if info["state"] == "RUNNING"
                )
            return {"status": "ok", "active_children": active, "target_children": self.config.num_children}

        return app

    async def _spawn_all_children(self) -> None:
        """Spawn all child serve jobs."""
        for i in range(self.config.num_children):
            port = self.config.child_spec.port_start + i
            try:
                await self._spawn_child(port)
            except Exception as e:
                logger.error(f"Failed to spawn child on port {port}: {e}")

    async def _spawn_child(self, port: int) -> None:
        """Spawn a single child serve job on the specified port.

        PA automatically assigns a job index. The job is tracked by port number.

        Args:
            port: Port number for the serve job.
        """
        import time

        # Build serve command
        serve_cmd = (
            f"CUDA_VISIBLE_DEVICES=0 python -m place_layers "
            f"+command=serve "
            f"+backend=transformers "
            f"+model={self.config.model_name} "
            f"port={port}"
        )

        resources = {
            "cpu": self.config.child_spec.cpu,
            "mem": self.config.child_spec.mem,
            "gpu": self.config.child_spec.gpu,
        }

        # Add gpuMem if specified (API expects camelCase)
        if self.config.child_spec.gpu_mem is not None:
            resources["gpuMem"] = self.config.child_spec.gpu_mem

        job_spec = {
            "name": f"serve-port-{port}",
            "image": self.config.child_spec.image,
            "resources": resources,
            "command": ["bash", "-c", serve_cmd],
            "preemptable": True,
            "restartable": True,
        }

        # Add data volumes if specified
        if self.config.child_spec.data:
            job_spec["data"] = self.config.child_spec.data

        logger.info(f"Spawning child on port {port} with command: {serve_cmd}")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{PA_BASE_URL}/jobs",
                json=[job_spec],
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            if not data or len(data) == 0:
                raise RuntimeError(f"PA returned no jobs for port {port}")

            job_spec_response = data[0]
            pa_job_index = job_spec_response.get("index")  # PA-assigned index

            # Find which orchestrator index this port corresponds to
            child_index = (port - self.config.child_spec.port_start)
            assert child_index == pa_job_index

            # Update spawn time for this port
            # job_id, IP, and state will be discovered by health monitor when polling PA API
            async with self._lock:
                self._index_info[child_index] = {
                    "port": port,
                    "current_job_id": None,  # Latest job ID for this index
                    "ip": None,  # Current IP address
                    "state": None,  # Current state
                    "failure_count": 0,  # Count of FAILED/SUCCEEDED (not CANCELLED)
                    "last_spawn_time": time.time()
                }
                logger.info(f"Spawned child on port {port}: PA assigned job_index={pa_job_index}")

    async def _health_monitor_loop(self) -> None:
        """Background task to monitor child health and restart failed jobs."""
        await asyncio.sleep(5)  # Wait for initial spawn

        while True:
            try:
                await self._check_and_restart_failed_children()
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")

            # Check every N seconds
            await asyncio.sleep(self.config.health_check_interval)

    async def _check_and_restart_failed_children(self) -> None:
        """Check child job status and restart failed ones.

        Simple algorithm:
        1. Get all jobs from PA API
        2. For each index (0 to num_children-1):
           a. Find the current job for this index
           b. If job is FAILED/SUCCEEDED: increment failure count, respawn if under limit
           c. If job is CANCELLED: respawn without counting
           d. If no job found: respawn
        """
        import time

        # Get all jobs from PA
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{PA_BASE_URL}", timeout=10.0)
            response.raise_for_status()
            stats = response.json()
            jobs = stats.get("jobs", [])

        async with self._lock:
            # Process each index
            # active jobs no more than total children count
            assert self.config.num_children >= len(self._index_info)
            # run trough jobs which have been already spawned
            for index in range(len(self._index_info)):
                info = self._index_info[index]
                current_job = jobs[index]
                
                old_state = info["state"]
                new_state = current_job.get("state", "UNKNOWN")
                info["state"] = new_state
                info["ip"] = current_job.get("ip")

                if new_state != old_state:
                    logger.info(f"Child {index}: {old_state} -> {new_state}")

                # Decide if we need to restart
                port = info["port"]
                if info["state"] in {"FAILED", "SUCCEEDED"}:
                    # Check retry limit
                    if info["failure_count"] < self.config.max_retries:
                        info["failure_count"] += 1
                        logger.info(
                            f"Child {index} (port {port}) {info['state']}, restarting "
                            f"(attempt {info['failure_count']}/{self.config.max_retries})"
                        )
                        await self._restart_child(index)
                    else:
                        logger.error(
                            f"Child {index} (port {port}) exceeded max retries ({self.config.max_retries}), "
                            f"not restarting"
                        )

                elif info["state"] in {"CANCELED", "INTERRUPTED"}:
                    # Restart without counting (unlimited retries for manual cancellation or preemption)
                    logger.info(f"Child {index} (port {port}) cancelled or preempted, restarting")
                    await self._restart_child(index)

    async def _restart_child(self, index: int) -> None:
        """Restart a child job at the given index using PA's rerun endpoint.

        Args:
            index: The job index to restart (0 to num_children-1).
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{PA_BASE_URL}/jobs/{index}/rerun",
                    timeout=30.0,
                )
                response.raise_for_status()

            logger.info(f"Successfully restarted child at index {index}")

        except Exception as e:
            logger.error(f"Failed to restart child at index {index}: {e}")
            raise


class PAOrchestratorCommandConfig(BaseCommandConfig):
    """Configuration for pa_orchestrator command."""

    _command_class: ClassVar[type[BaseCommand]] = PAOrchestratorCommand

    command: Literal["pa_orchestrator"] = "pa_orchestrator"

    # Model to serve (e.g., "apriel2_0_5b_dev")
    model_name: str = Field(description="Model config name to use for serve jobs")

    # Child job specification
    child_spec: PAChildSpec = Field(description="Specification for child serve jobs")

    # Number of child jobs to maintain
    num_children: int = Field(description="Number of child serve jobs", ge=1)

    # Health check interval
    health_check_interval: int = Field(
        default=30, description="Health check interval in seconds", ge=1
    )

    # Minimum time between spawns for same child index (rate limiting)
    min_spawn_interval: int = Field(
        default=60,
        description="Minimum seconds between spawn attempts for same child index",
        ge=1,
    )

    # Maximum retries for FAILED/SUCCEEDED jobs (CANCELLED has unlimited retries)
    max_retries: int = Field(
        default=5,
        description="Maximum retry attempts for failed/succeeded jobs per child",
        ge=0,
    )

    # Server bind address
    host: str = Field(
        default="0.0.0.0",
        description="Host address to bind to",
    )

    # Server port
    port: int = Field(
        default=8080, description="Port number to listen on", ge=1, le=65535
    )
