"""SA event types for annealing search.

Self-contained event models emitted by ``run_multi_start_sa`` and consumed
by any observer that implements the ``SAObserver`` protocol (including
``SearchObserver`` from the progress subsystem, which re-exports these types).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class AnnealingStepEvent(BaseModel):
    """Emitted after each annealing step.

    Contains full state information for analysis and replay.
    """

    model_config = {"frozen": True}

    event_type: Literal["annealing_step"] = "annealing_step"
    step: int  # 0-based step number
    max_steps: int
    restart_index: int  # Which restart this step belongs to
    perturbation_size: int  # Number of positions perturbed
    move_type: Literal["random_ruin", "contiguous_ruin", "type_mutation"]
    proposed_assignments: tuple[str, ...]  # Placement after move
    proposed_score: float
    temperature: float  # Temperature before this step
    delta: float  # proposed_score - current_score
    acceptance_probability: float  # exp(delta/T) if delta < 0, else 1.0
    accepted: bool  # Whether the move was accepted
    current_assignments: tuple[str, ...]  # Current state after decision
    current_score: float  # Current score after decision
    best_assignments: tuple[str, ...]  # Best seen so far
    best_score: float  # Best score seen so far


class AnnealingCompletedEvent(BaseModel):
    """Emitted when annealing within an allocation completes."""

    model_config = {"frozen": True}

    event_type: Literal["annealing_completed"] = "annealing_completed"
    steps_taken: int
    final_temperature: float
    termination_reason: Literal["max_steps", "min_temperature", "no_swaps"]
    best_assignments: tuple[str, ...]
    best_score: float
    restarts_completed: int
    best_restart: int
    total_reheats: int


class AnnealingReheatEvent(BaseModel):
    """Emitted when adaptive reheating triggers during annealing."""

    model_config = {"frozen": True}

    event_type: Literal["annealing_reheat"] = "annealing_reheat"
    restart_index: int
    step: int
    old_temperature: float
    new_temperature: float
    acceptance_rate: float


class AnnealingRestartEvent(BaseModel):
    """Emitted when a new SA restart begins."""

    model_config = {"frozen": True}

    event_type: Literal["annealing_restart"] = "annealing_restart"
    restart_index: int
    num_restarts: int
    steps_budget: int
    initial_assignments: tuple[str, ...]
