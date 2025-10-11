"""Run request/result contracts used by runners and the facade."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from enum import Enum
from .io import DataRef, SecretRef, ExecutionContext

@dataclass
class EvaluationSpec:
    """Describe evaluation to compute as part of a run result.

    Attributes:
        metric_ids: List of metric ids to compute.
        y_true_key: Which input key contains the ground truth data.
        y_pred_source: Reserved for future use; which result field to read.
        compute_train/compute_valid: Whether to compute train/validation splits.
        train_indices/test_indices: Optional index lists to slice inputs/outputs.
        bindings: Map of metric_id to binding used to compute it.
    """
    metric_ids: List[str]
    y_true_key: str = "y"
    y_pred_source: str = "predictions"
    compute_train: bool = True
    compute_valid: bool = True
    train_indices: Optional[List[int]] = None
    test_indices: Optional[List[int]] = None
    bindings: Dict[str, Any] = field(default_factory=dict)  # metric_id -> ExecutionBinding

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON‑serializable dictionary for audit/logging."""
        return {
            "metric_ids": self.metric_ids,
            "y_true_key": self.y_true_key,
            "y_pred_source": self.y_pred_source,
            "compute_train": self.compute_train,
            "compute_valid": self.compute_valid,
            "train_indices": self.train_indices,
            "test_indices": self.test_indices,
            "bindings": {k: (None if v is None else v.to_dict()) for k, v in self.bindings.items()},
        }

class RunStatus(Enum):
    """Lifecycle states for a model run."""
    QUEUED = "queued"
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ModelRunRequest:
    """Request to execute a model with inputs/params and optional evaluation."""
    model_id: str
    inputs: Dict[str, DataRef]
    params: Dict[str, Any]
    context: ExecutionContext = field(default_factory=ExecutionContext)
    dataset_version: Optional[str] = None
    secrets: list[SecretRef] = field(default_factory=list)
    binding: Optional[Any] = None  # ExecutionBinding (kept outside registries)
    evaluation: Optional[EvaluationSpec] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON‑serializable dictionary for audit/logging."""
        return {
            "model_id": self.model_id,
            "inputs": {k: v.to_dict() for k, v in self.inputs.items()},
            "params": self.params,
            "context": {"random_seed": self.context.random_seed, "timeout_s": self.context.timeout_s,
                        "device": self.context.device, "resources": self.context.resources, "tags": self.context.tags},
            "dataset_version": self.dataset_version,
            "secrets": [vars(s) for s in self.secrets],
            "binding": None if self.binding is None else self.binding.to_dict(),
            "evaluation": None if self.evaluation is None else self.evaluation.to_dict(),
        }

@dataclass
class ModelRunHandle:
    """Lightweight handle returned on submission to track a run."""
    run_id: str
    status: RunStatus

@dataclass
class ModelRunResult:
    """Result for a model run (partial or final)."""
    run_id: str
    status: RunStatus
    predictions: Optional[DataRef] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, DataRef] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    eval_train: Dict[str, float] = field(default_factory=dict)
    eval_valid: Dict[str, float] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MetricComputeRequest:
    """Request to compute a metric given references to data."""
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
        """Serialize request into a JSON‑serializable dictionary."""
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
    """Result of a metric computation."""
    score: Optional[float]
    artifacts: Dict[str, DataRef] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
