"""Rich progress bar utilities for evaluation loops.

This module provides a shared Rich console and progress bar factory for
consistent progress reporting across evaluation commands. The key insight
is that Rich's Progress works seamlessly with RichHandler logging when
they share the same Console instance - logs appear above the progress bar.

Usage:
    from place_layers.evaluator.progress import create_progress

    with create_progress() as progress:
        task_id = progress.add_task("Processing", total=100)
        for item in items:
            # ... process item ...
            progress.advance(task_id)

The progress bar is visible on the console while detailed logs go to file.
"""

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Shared console instance - use this for any RichHandler to ensure
# progress bars and logs render correctly together
console = Console()


def create_progress(transient: bool = False) -> Progress:
    """Create a Progress instance with standard columns.

    Args:
        transient: If True, progress bar disappears after completion.
                   If False (default), it remains visible.

    Returns:
        Configured Progress instance.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=transient,
    )
