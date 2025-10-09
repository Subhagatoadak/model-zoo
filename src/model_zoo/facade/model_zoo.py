from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from ..registry.models import ModelRegistry, ModelSpec
from ..registry.metrics import MetricRegistry, EvalMetricSpec
from ..contracts.validation import ParamValidationService
from ..contracts.audit import AuditLog, AuditEvent
from ..contracts.run_requests import (
    ModelRunRequest, ModelRunHandle, ModelRunResult,
    MetricComputeRequest, MetricComputeResult
)
from ..runners.base import ModelRunner, MetricRunner

@dataclass
class ModelZoo:
    model_registry: ModelRegistry
    metric_registry: MetricRegistry
    model_runner: ModelRunner
    metric_runner: MetricRunner
    validator: ParamValidationService
    audit: Optional[AuditLog] = None
    binding_registry: Optional[Any] = None  # optional external binding store

    # registries
    def register_model(self, spec: ModelSpec, actor: str = "system") -> None:
        self.model_registry.register(spec)
        self._audit("REGISTER_MODEL", actor, spec.model_id, spec.to_dict())

    def register_metric(self, spec: EvalMetricSpec, actor: str = "system") -> None:
        self.metric_registry.register(spec)
        self._audit("REGISTER_METRIC", actor, spec.metric_id, spec.to_dict())

    def list_models(self) -> List[ModelSpec]:
        return self.model_registry.list()

    def list_metrics(self) -> List[EvalMetricSpec]:
        return self.metric_registry.list()

    # runs
    def submit_run(self, req: ModelRunRequest, actor: str = "system") -> ModelRunHandle:
        spec = self.model_registry.get(req.model_id)
        self.validator.validate_model_params(spec, req.params)
        # If a binding registry exists, attach binding here (kept outside registries)
        self._audit("RUN_SUBMIT", actor, req.model_id, req.to_dict())
        return self.model_runner.submit(req)

    def get_run_result(self, run_id: str, actor: str = "system") -> ModelRunResult:
        res = self.model_runner.result(run_id)
        self._audit("RUN_RESULT", actor, run_id, {"status": res.status.value})
        return res

    def compute_metric(self, req: MetricComputeRequest, actor: str = "system") -> MetricComputeResult:
        spec = self.metric_registry.get(req.metric_id)
        self.validator.validate_metric_params(spec, req.params)
        self._audit("METRIC_COMPUTE", actor, req.metric_id, req.to_dict())
        return self.metric_runner.compute(req)

    def _audit(self, kind: str, who: str, ref_id: str, payload: Dict[str, Any]) -> None:
        if not self.audit: return
        from datetime import datetime, timezone
        evt = AuditEvent(kind=kind, who=who, when=datetime.now(timezone.utc).isoformat(), ref_id=ref_id, payload=payload)
        self.audit.append(evt)
