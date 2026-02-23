"""Abstract base for surrogate models.

Defines the ``BaseSurrogate`` interface that all surrogates must implement,
and the ``SurrogateDiagnostics`` base return type for surrogate-specific
quality metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict


class SurrogateDiagnostics(BaseModel, ABC):
    """Base return type for surrogate diagnostics. Subclass per surrogate."""

    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def summary(self) -> tuple[str, float]:
        """Return (label, value) for UI display."""


class FrequentistCVDiagnostics(SurrogateDiagnostics):
    """Diagnostics from frequentist k-fold cross-validation."""

    spearman_cv: float

    def summary(self) -> tuple[str, float]:
        return ("\u03c1_cv", self.spearman_cv)


class BaseSurrogate(ABC):
    """Abstract base for surrogate models."""

    @abstractmethod
    def fit(
        self,
        assignments: Sequence[Sequence[str]],
        y: np.ndarray,
    ) -> None: ...

    @abstractmethod
    def predict(
        self, assignments: Sequence[Sequence[str]],
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @abstractmethod
    def predict_mean(
        self, assignments: Sequence[Sequence[str]],
    ) -> np.ndarray: ...

    @abstractmethod
    def predict_mean_one(
        self, assignment: Sequence[str],
    ) -> float: ...

    @abstractmethod
    def diagnostics(
        self,
        assignments: Sequence[Sequence[str]],
        y: np.ndarray,
        *,
        folds: int | None,
        rng_seed: int = 0,
    ) -> SurrogateDiagnostics:
        """Surrogate-specific diagnostic payload."""
