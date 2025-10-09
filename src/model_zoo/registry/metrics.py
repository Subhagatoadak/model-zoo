from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
from enum import Enum
from ..contracts.hyperparams import HyperParamSpace

class MetricDirection(Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"

class MetricKind(Enum):
    API = "api"
    LIBRARY = "library"
    PIPELINE = "pipeline"

@dataclass
class EvalMetricSpec:
    metric_id: str
    name: str
    version: str
    direction: MetricDirection
    kind: MetricKind
    params: HyperParamSpace = field(default_factory=HyperParamSpace)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "name": self.name,
            "version": self.version,
            "direction": self.direction.value,
            "kind": self.kind.value,
            "params_schema": self.params.to_json_schema(),
            "tags": self.tags,
            "metadata": self.metadata,
        }

class MetricRegistry:
    def __init__(self) -> None:
        self._metrics: Dict[str, EvalMetricSpec] = {}

    def register(self, spec: EvalMetricSpec) -> None:
        if spec.metric_id in self._metrics:
            raise KeyError(f"Metric '{spec.metric_id}' already exists. Use replace().")
        self._metrics[spec.metric_id] = spec

    def replace(self, metric_id: str, spec: EvalMetricSpec) -> None:
        self._metrics[metric_id] = spec

    def remove(self, metric_id: str) -> None:
        self._metrics.pop(metric_id, None)

    def get(self, metric_id: str) -> EvalMetricSpec:
        if metric_id not in self._metrics:
            raise KeyError(f"Unknown metric_id: {metric_id}")
        return self._metrics[metric_id]

    def list(self) -> List[EvalMetricSpec]:
        return list(self._metrics.values())
