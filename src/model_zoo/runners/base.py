from __future__ import annotations
from abc import ABC, abstractmethod
from ..contracts.run_requests import (
    ModelRunRequest, ModelRunHandle, ModelRunResult,
    MetricComputeRequest, MetricComputeResult
)

class ModelRunner(ABC):
    @abstractmethod
    def submit(self, req: ModelRunRequest) -> ModelRunHandle: ...
    @abstractmethod
    def status(self, run_id: str): ...
    @abstractmethod
    def result(self, run_id: str) -> ModelRunResult: ...
    @abstractmethod
    def cancel(self, run_id: str) -> None: ...

class MetricRunner(ABC):
    @abstractmethod
    def compute(self, req: MetricComputeRequest) -> MetricComputeResult: ...
