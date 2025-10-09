from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class ModelCapability:
    supports_proba: bool = False
    supports_samples: bool = False
    task_types: List[str] = field(default_factory=list)
    input_formats: List[str] = field(default_factory=list)
    output_formats: List[str] = field(default_factory=list)
    max_batch_size: Optional[int] = None
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supports_proba": self.supports_proba, "supports_samples": self.supports_samples,
            "task_types": self.task_types, "input_formats": self.input_formats, "output_formats": self.output_formats,
            "max_batch_size": self.max_batch_size, "notes": self.notes
        }

@dataclass
class MetricRequirement:
    requires_y_true: bool = True
    requires_y_pred: bool = True
    accepts_proba: bool = False
    accepts_samples: bool = False
    task_types: List[str] = field(default_factory=list)
    extra_payload_spec: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requires_y_true": self.requires_y_true, "requires_y_pred": self.requires_y_pred,
            "accepts_proba": self.accepts_proba, "accepts_samples": self.accepts_samples,
            "task_types": self.task_types, "extra_payload_spec": self.extra_payload_spec
        }

class CompatibilityService:
    @staticmethod
    def ensure_compatible(model_cap: ModelCapability, metric_req: MetricRequirement) -> None:
        raise NotImplementedError("Implement compatibility checks in your platform.")
