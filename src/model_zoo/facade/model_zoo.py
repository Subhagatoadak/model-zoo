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

    def compute_all_metrics(
        self,
        *,
        y_true: Any,
        y_pred: Any = None,
        proba: Any = None,
        samples: Any = None,
        metrics: Optional[list[str]] = None,
        bindings: Optional[dict[str, Any]] = None,
        params_by_metric: Optional[dict[str, dict[str, Any]]] = None,
        actor: str = "system",
    ) -> dict[str, MetricComputeResult]:
        """Compute a set of registered metrics in one call.

        - metrics: optional list of metric_ids to compute; defaults to all registered.
        - bindings: map metric_id -> ExecutionBinding. If not provided, requires a binding registry.
        - params_by_metric: map metric_id -> params dict overrides.
        Returns a dict metric_id -> MetricComputeResult.
        """
        params_by_metric = params_by_metric or {}
        bindings = bindings or {}

        # Resolve the metric set
        all_specs = {m.metric_id: m for m in self.metric_registry.list()}
        metric_ids = metrics if metrics is not None else list(all_specs.keys())

        results: dict[str, MetricComputeResult] = {}
        for mid in metric_ids:
            if mid not in all_specs:
                raise KeyError(f"Unknown metric_id: {mid}")
            spec = all_specs[mid]

            # Resolve binding per metric
            binding = bindings.get(mid)
            if binding is None and self.binding_registry is not None:
                # If a binding registry is provided by the app, try to fetch from it.
                try:
                    binding = self.binding_registry.get_metric_binding(mid)
                except Exception:
                    binding = None
            if binding is None:
                raise ValueError(f"No binding provided for metric '{mid}' and no binding registry configured")

            def to_ref(v):
                # Allow passing either DataRef or raw inline values; wrap raw values as DataRef(INLINE)
                from ..contracts.io import DataRef, DataRefType
                if isinstance(v, DataRef):
                    return v
                return DataRef(ref_type=DataRefType.INLINE, inline=v)

            req = MetricComputeRequest(
                metric_id=mid,
                direction=spec.direction.value,
                y_true=to_ref(y_true),
                y_pred=None if y_pred is None else to_ref(y_pred),
                proba=None if proba is None else to_ref(proba),
                samples=None if samples is None else to_ref(samples),
                params=params_by_metric.get(mid, {}),
                binding=binding,
            )
            results[mid] = self.compute_metric(req, actor=actor)
        return results

    def _audit(self, kind: str, who: str, ref_id: str, payload: Dict[str, Any]) -> None:
        if not self.audit: return
        from datetime import datetime, timezone
        evt = AuditEvent(kind=kind, who=who, when=datetime.now(timezone.utc).isoformat(), ref_id=ref_id, payload=payload)
        self.audit.append(evt)
