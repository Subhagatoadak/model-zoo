from __future__ import annotations
"""Abstract runner interfaces used by the Model Zoo facade.

Runners are responsible for execution only. They do not perform registration or
parameter validation — those are handled upstream by the facade/registries.
"""
from abc import ABC, abstractmethod
from ..contracts.run_requests import (
    ModelRunRequest, ModelRunHandle, ModelRunResult,
    MetricComputeRequest, MetricComputeResult
)

class ModelRunner(ABC):
    """Executes model runs according to a provided binding.

    Concrete implementations may execute locally (in‑process), over HTTP, or via
    an orchestrator. Implementations should be idempotent and thread‑safe where
    feasible.
    """

    @abstractmethod
    def submit(self, req: ModelRunRequest) -> ModelRunHandle:
        """Submit a run and return a handle for tracking.

        Implementations may start execution immediately or enqueue work.
        """

    @abstractmethod
    def status(self, run_id: str):
        """Return current status for a run identifier."""

    @abstractmethod
    def result(self, run_id: str) -> ModelRunResult:
        """Return the current or final result for a run.

        If the run is still in progress, a partial result may be returned.
        """

    @abstractmethod
    def cancel(self, run_id: str) -> None:
        """Request cancellation of an in‑flight run (best effort)."""

class MetricRunner(ABC):
    """Computes evaluation metrics given references to y_true/predictions/etc."""

    @abstractmethod
    def compute(self, req: MetricComputeRequest) -> MetricComputeResult:
        """Compute a metric and return a score and optional artifacts."""
