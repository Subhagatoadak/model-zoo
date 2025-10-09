from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum
from .io import DataRef, SecretRef, ExecutionContext

class RunStatus(Enum):
    QUEUED = "queued"
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ModelRunRequest:
    model_id: str
    inputs: Dict[str, DataRef]
    params: Dict[str, Any]
    context: ExecutionContext = field(default_factory=ExecutionContext)
    dataset_version: Optional[str] = None
    secrets: list[SecretRef] = field(default_factory=list)
    binding: Optional[Any] = None  # ExecutionBinding (kept outside registries)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "inputs": {k: v.to_dict() for k, v in self.inputs.items()},
            "params": self.params,
            "context": {"random_seed": self.context.random_seed, "timeout_s": self.context.timeout_s,
                        "device": self.context.device, "resources": self.context.resources, "tags": self.context.tags},
            "dataset_version": self.dataset_version,
            "secrets": [vars(s) for s in self.secrets],
            "binding": None if self.binding is None else self.binding.to_dict(),
        }

@dataclass
class ModelRunHandle:
    run_id: str
    status: RunStatus

@dataclass
class ModelRunResult:
    run_id: str
    status: RunStatus
    predictions: Optional[DataRef] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, DataRef] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None

@dataclass
class MetricComputeRequest:
    metric_id: str
    direction: str
    y_true: Optional[DataRef] = None
    y_pred: Optional[DataRef] = None
    proba: Optional[DataRef] = None
    samples: Optional[DataRef] = None
    params: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    secrets: list[SecretRef] = field(default_factory=list)
    binding: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        def enc(x): return None if x is None else x.to_dict()
        return {
            "metric_id": self.metric_id, "direction": self.direction,
            "y_true": enc(self.y_true), "y_pred": enc(self.y_pred),
            "proba": enc(self.proba), "samples": enc(self.samples),
            "params": self.params, "extra": self.extra,
            "secrets": [vars(s) for s in self.secrets],
            "binding": None if self.binding is None else self.binding.to_dict(),
        }

@dataclass
class MetricComputeResult:
    score: Optional[float]
    artifacts: Dict[str, DataRef] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
