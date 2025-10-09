from __future__ import annotations
"""Metric registry primitives.

Defines the declarative `EvalMetricSpec` along with an in‑memory `MetricRegistry`.
Registries are pure metadata; execution is handled via bindings and runners.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List
from enum import Enum
from ..contracts.hyperparams import HyperParamSpace

class MetricDirection(Enum):
    """Indicates how to optimize a metric.

    - HIGHER_IS_BETTER: e.g., accuracy, R²
    - LOWER_IS_BETTER: e.g., MSE, MAE
    """

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"

class MetricKind(Enum):
    """Mechanism used to compute a metric (mirrors `ModelKind`)."""

    API = "api"
    LIBRARY = "library"
    PIPELINE = "pipeline"

@dataclass
class EvalMetricSpec:
    """Declarative specification for a metric in the registry.

    Attributes:
        metric_id: Unique identifier for the metric.
        name: Human‑readable display name.
        version: Revision string for the metric spec.
        direction: Optimization direction (see `MetricDirection`).
        kind: Execution mechanism (see `MetricKind`).
        params: Optional hyperparameter schema for the metric.
        tags: User‑defined labels for discovery/organization.
        metadata: Arbitrary extra metadata for tooling/integration.
    """
    metric_id: str
    name: str
    version: str
    direction: MetricDirection
    kind: MetricKind
    params: HyperParamSpace = field(default_factory=HyperParamSpace)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON‑serializable dictionary for logs/APIs."""
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
    """In‑memory registry for metric specifications."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._metrics: Dict[str, EvalMetricSpec] = {}

    def register(self, spec: EvalMetricSpec) -> None:
        """Register a new metric spec or raise if it exists."""
        if spec.metric_id in self._metrics:
            raise KeyError(f"Metric '{spec.metric_id}' already exists. Use replace().")
        self._metrics[spec.metric_id] = spec

    def replace(self, metric_id: str, spec: EvalMetricSpec) -> None:
        """Replace or upsert a metric spec under `metric_id`."""
        self._metrics[metric_id] = spec

    def remove(self, metric_id: str) -> None:
        """Remove a metric by id. Does nothing if absent."""
        self._metrics.pop(metric_id, None)

    def get(self, metric_id: str) -> EvalMetricSpec:
        """Fetch a metric spec by id or raise KeyError if missing."""
        if metric_id not in self._metrics:
            raise KeyError(f"Unknown metric_id: {metric_id}")
        return self._metrics[metric_id]

    def list(self) -> List[EvalMetricSpec]:
        """Return all registered metric specs as a list."""
        return list(self._metrics.values())
