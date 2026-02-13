"""Memory logging utilities for debugging OOM issues.

Provides comprehensive GPU and CPU memory observability with logging
at key points during trace generation.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Callable

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MemoryStats(BaseModel):
    """Snapshot of memory usage.

    On shared memory architectures (unified memory), GPU and CPU memory
    are part of the same pool, so tracking both is critical.
    """

    # GPU memory (bytes)
    gpu_allocated: int
    gpu_reserved: int
    gpu_max_allocated: int
    gpu_total: int

    # Process memory (bytes)
    cpu_rss: int  # Resident Set Size - actual physical memory used
    cpu_vms: int  # Virtual Memory Size - total virtual address space
    cpu_shared: int  # Shared memory (e.g., shared libraries)
    cpu_data: int  # Data segment size (heap + stack)

    # System memory (bytes)
    sys_total: int
    sys_available: int
    sys_used: int
    sys_swap_total: int
    sys_swap_used: int

    @property
    def gpu_allocated_gb(self) -> float:
        return self.gpu_allocated / (1024**3)

    @property
    def gpu_reserved_gb(self) -> float:
        return self.gpu_reserved / (1024**3)

    @property
    def gpu_max_allocated_gb(self) -> float:
        return self.gpu_max_allocated / (1024**3)

    @property
    def gpu_total_gb(self) -> float:
        return self.gpu_total / (1024**3)

    @property
    def gpu_free_gb(self) -> float:
        return (self.gpu_total - self.gpu_reserved) / (1024**3)

    @property
    def cpu_rss_gb(self) -> float:
        return self.cpu_rss / (1024**3)

    @property
    def cpu_vms_gb(self) -> float:
        return self.cpu_vms / (1024**3)

    @property
    def sys_available_gb(self) -> float:
        return self.sys_available / (1024**3)

    @property
    def sys_used_gb(self) -> float:
        return self.sys_used / (1024**3)

    @property
    def sys_total_gb(self) -> float:
        return self.sys_total / (1024**3)

    @property
    def sys_swap_used_gb(self) -> float:
        return self.sys_swap_used / (1024**3)

    @property
    def total_process_memory_gb(self) -> float:
        """Total memory used by process (GPU + CPU RSS on shared memory)."""
        # On unified memory, GPU allocated is part of the same pool
        return (self.gpu_allocated + self.cpu_rss) / (1024**3)

    @property
    def memory_pressure_pct(self) -> float:
        """Memory pressure as percentage of system memory."""
        if self.sys_total == 0:
            return 0.0
        return 100.0 * (self.sys_total - self.sys_available) / self.sys_total


def get_memory_stats() -> MemoryStats:
    """Get current memory usage statistics.

    Returns:
        MemoryStats with GPU, process, and system memory information.
    """
    import torch

    # GPU memory
    gpu_allocated = 0
    gpu_reserved = 0
    gpu_max_allocated = 0
    gpu_total = 0

    if torch.cuda.is_available():
        try:
            gpu_allocated = torch.cuda.memory_allocated()
            gpu_reserved = torch.cuda.memory_reserved()
            gpu_max_allocated = torch.cuda.max_memory_allocated()
            gpu_total = torch.cuda.get_device_properties(0).total_memory
        except Exception as e:
            logger.warning(f"Failed to get GPU memory stats: {e}")

    # Process memory
    cpu_rss = 0
    cpu_vms = 0
    cpu_shared = 0
    cpu_data = 0

    # System memory
    sys_total = 0
    sys_available = 0
    sys_used = 0
    sys_swap_total = 0
    sys_swap_used = 0

    try:
        import psutil

        # Process memory
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        cpu_rss = mem_info.rss
        cpu_vms = mem_info.vms
        cpu_shared = getattr(mem_info, "shared", 0)
        cpu_data = getattr(mem_info, "data", 0)

        # System memory
        sys_mem = psutil.virtual_memory()
        sys_total = sys_mem.total
        sys_available = sys_mem.available
        sys_used = sys_mem.used

        # Swap memory
        swap_mem = psutil.swap_memory()
        sys_swap_total = swap_mem.total
        sys_swap_used = swap_mem.used

    except ImportError:
        # psutil not available, try /proc filesystem
        try:
            # Process memory from /proc/self/status
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        cpu_rss = int(line.split()[1]) * 1024  # KB to bytes
                    elif line.startswith("VmSize:"):
                        cpu_vms = int(line.split()[1]) * 1024
                    elif line.startswith("RssAnon:"):
                        cpu_data = int(line.split()[1]) * 1024
                    elif line.startswith("RssShmem:"):
                        cpu_shared = int(line.split()[1]) * 1024
        except Exception:
            pass

        try:
            # System memory from /proc/meminfo
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split()
                    if line.startswith("MemTotal:"):
                        sys_total = int(parts[1]) * 1024
                    elif line.startswith("MemAvailable:"):
                        sys_available = int(parts[1]) * 1024
                    elif line.startswith("MemFree:"):
                        # Fallback if MemAvailable not present
                        if sys_available == 0:
                            sys_available = int(parts[1]) * 1024
                    elif line.startswith("SwapTotal:"):
                        sys_swap_total = int(parts[1]) * 1024
                    elif line.startswith("SwapFree:"):
                        swap_free = int(parts[1]) * 1024
                        sys_swap_used = sys_swap_total - swap_free
            sys_used = sys_total - sys_available
        except Exception:
            pass

    except Exception as e:
        logger.warning(f"Failed to get memory stats: {e}")

    return MemoryStats(
        gpu_allocated=gpu_allocated,
        gpu_reserved=gpu_reserved,
        gpu_max_allocated=gpu_max_allocated,
        gpu_total=gpu_total,
        cpu_rss=cpu_rss,
        cpu_vms=cpu_vms,
        cpu_shared=cpu_shared,
        cpu_data=cpu_data,
        sys_total=sys_total,
        sys_available=sys_available,
        sys_used=sys_used,
        sys_swap_total=sys_swap_total,
        sys_swap_used=sys_swap_used,
    )


def log_memory(prefix: str = "", level: int = logging.INFO) -> MemoryStats:
    """Log current memory usage.

    Args:
        prefix: Optional prefix for the log message.
        level: Logging level (default INFO).

    Returns:
        MemoryStats for programmatic use.
    """
    stats = get_memory_stats()

    prefix_str = f"[{prefix}] " if prefix else ""

    # GPU line
    gpu_msg = (
        f"GPU: alloc={stats.gpu_allocated_gb:.2f}GB, "
        f"reserved={stats.gpu_reserved_gb:.2f}GB, "
        f"peak={stats.gpu_max_allocated_gb:.2f}GB, "
        f"free={stats.gpu_free_gb:.2f}GB"
    )

    # Process line (critical for shared memory)
    proc_msg = (
        f"Process: RSS={stats.cpu_rss_gb:.2f}GB, "
        f"VMS={stats.cpu_vms_gb:.2f}GB"
    )

    # System line (shows overall pressure)
    sys_msg = (
        f"System: used={stats.sys_used_gb:.2f}/{stats.sys_total_gb:.2f}GB "
        f"({stats.memory_pressure_pct:.1f}% pressure), "
        f"avail={stats.sys_available_gb:.2f}GB"
    )

    # Swap line (if any swap is used, it's a warning sign)
    swap_msg = ""
    if stats.sys_swap_used > 0:
        swap_msg = f", SWAP={stats.sys_swap_used_gb:.2f}GB USED!"

    msg = f"{prefix_str}Memory | {gpu_msg} | {proc_msg} | {sys_msg}{swap_msg}"
    logger.log(level, msg)

    # Log warning if memory pressure is high
    if stats.memory_pressure_pct > 85:
        logger.warning(
            f"{prefix_str}HIGH MEMORY PRESSURE: {stats.memory_pressure_pct:.1f}% "
            f"(only {stats.sys_available_gb:.2f}GB available)"
        )

    # Log warning if swap is being used
    if stats.sys_swap_used > 0:
        logger.warning(
            f"{prefix_str}SWAP IN USE: {stats.sys_swap_used_gb:.2f}GB - "
            f"performance will be severely degraded!"
        )

    return stats


def log_memory_delta(
    before: MemoryStats, after: MemoryStats, prefix: str = ""
) -> None:
    """Log the change in memory usage between two snapshots.

    Args:
        before: Memory stats before operation.
        after: Memory stats after operation.
        prefix: Optional prefix for the log message.
    """
    gpu_delta = (after.gpu_allocated - before.gpu_allocated) / (1024**3)
    cpu_delta = (after.cpu_rss - before.cpu_rss) / (1024**3)
    vms_delta = (after.cpu_vms - before.cpu_vms) / (1024**3)
    sys_avail_delta = (after.sys_available - before.sys_available) / (1024**3)

    prefix_str = f"[{prefix}] " if prefix else ""

    def fmt(val: float) -> str:
        sign = "+" if val >= 0 else ""
        return f"{sign}{val:.2f}GB"

    logger.info(
        f"{prefix_str}Memory delta: GPU {fmt(gpu_delta)}, "
        f"RSS {fmt(cpu_delta)}, VMS {fmt(vms_delta)}, "
        f"sys_avail {fmt(sys_avail_delta)}"
    )


def reset_peak_memory_stats() -> None:
    """Reset GPU peak memory tracking."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        logger.info("Reset GPU peak memory stats")


def force_garbage_collection() -> None:
    """Force garbage collection and CUDA cache clearing."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    logger.info("Forced garbage collection and CUDA cache clear")


def log_gpu_memory_summary() -> None:
    """Log detailed GPU memory summary from PyTorch."""
    import torch

    if torch.cuda.is_available():
        try:
            summary = torch.cuda.memory_summary(abbreviated=True)
            logger.info(f"GPU Memory Summary:\n{summary}")
        except Exception as e:
            logger.warning(f"Failed to get GPU memory summary: {e}")


class MemoryMonitor:
    """Context manager for monitoring memory usage during an operation."""

    def __init__(self, name: str, log_on_enter: bool = True, log_on_exit: bool = True):
        """Initialize memory monitor.

        Args:
            name: Name of the operation being monitored.
            log_on_enter: Whether to log memory on context entry.
            log_on_exit: Whether to log memory and delta on exit.
        """
        self.name = name
        self.log_on_enter = log_on_enter
        self.log_on_exit = log_on_exit
        self.before: MemoryStats | None = None
        self.after: MemoryStats | None = None

    def __enter__(self) -> "MemoryMonitor":
        self.before = get_memory_stats()
        if self.log_on_enter:
            log_memory(f"{self.name} START")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.after = get_memory_stats()

        if exc_type is not None:
            # An exception occurred - log memory state for debugging
            logger.error(
                f"[{self.name} EXCEPTION] {exc_type.__name__}: {exc_val}"
            )
            log_memory(f"{self.name} EXCEPTION", level=logging.ERROR)

            # Check if it's an OOM error
            if _is_oom_error(exc_type, exc_val):
                logger.error(f"[{self.name}] OUT OF MEMORY DETECTED!")
                log_gpu_memory_summary()

        elif self.log_on_exit:
            log_memory(f"{self.name} END")
            if self.before is not None:
                log_memory_delta(self.before, self.after, f"{self.name} DELTA")


def _is_oom_error(exc_type, exc_val) -> bool:
    """Check if an exception is an OOM error."""
    if exc_type is None:
        return False

    # CUDA OOM
    if "CUDA out of memory" in str(exc_val):
        return True
    if "OutOfMemoryError" in exc_type.__name__:
        return True

    # CPU OOM
    if "Cannot allocate memory" in str(exc_val):
        return True
    if isinstance(exc_val, MemoryError):
        return True

    return False


def wrap_with_memory_logging(
    func: Callable, name: str | None = None
) -> Callable:
    """Decorator to add memory logging around a function.

    Args:
        func: Function to wrap.
        name: Name for logging (defaults to function name).

    Returns:
        Wrapped function with memory logging.
    """
    import functools

    op_name = name or func.__name__

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with MemoryMonitor(op_name):
            return func(*args, **kwargs)

    return wrapper


def log_tensor_info(tensor, name: str = "tensor") -> None:
    """Log information about a tensor including memory footprint.

    Args:
        tensor: PyTorch tensor to inspect.
        name: Name for the tensor in logs.
    """
    import torch

    if not isinstance(tensor, torch.Tensor):
        logger.warning(f"{name} is not a tensor: {type(tensor)}")
        return

    numel = tensor.numel()
    dtype_size = tensor.element_size()
    memory_bytes = numel * dtype_size
    memory_mb = memory_bytes / (1024**2)

    logger.info(
        f"Tensor '{name}': shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"device={tensor.device}, memory={memory_mb:.2f}MB"
    )
