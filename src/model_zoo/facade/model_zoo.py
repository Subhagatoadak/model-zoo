from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from ..registry.models import ModelRegistry, ModelSpec
from ..registry.metrics import MetricRegistry, EvalMetricSpec
from ..contracts.validation import ParamValidationService
from ..contracts.audit import AuditLog, AuditEvent
from ..contracts.run_requests import (
    ModelRunRequest, ModelRunHandle, ModelRunResult,
    MetricComputeRequest, MetricComputeResult, EvaluationSpec
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
    _eval_specs: Dict[str, Dict[str, Any]] = None  # internal: run_id -> {spec, y_true_ref}

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
        handle = self.model_runner.submit(req)
        # Initialize eval spec store
        if self._eval_specs is None:
            self._eval_specs = {}
        if req.evaluation is not None:
            y_key = req.evaluation.y_true_key
            y_ref = req.inputs.get(y_key)
            self._eval_specs[handle.run_id] = {"spec": req.evaluation, "y_true": y_ref}
        return handle

    def get_run_result(self, run_id: str, actor: str = "system") -> ModelRunResult:
        res = self.model_runner.result(run_id)
        self._audit("RUN_RESULT", actor, run_id, {"status": res.status.value})
        # Inline evaluation if an evaluation spec exists for this run and predictions are available
        if getattr(self, "_eval_specs", None) and run_id in self._eval_specs:
            plan = self._eval_specs[run_id]
            spec: EvaluationSpec = plan["spec"]
            y_true_ref = plan.get("y_true")
            if res.status.name == "SUCCEEDED" and res.predictions is not None and y_true_ref is not None:
                # Only compute if not already computed
                if not res.eval_scores:
                    try:
                        # Prefer inline lists for demo evaluation
                        if y_true_ref.ref_type.name == "INLINE" and res.predictions.ref_type.name == "INLINE":
                            yt = y_true_ref
                            yp = res.predictions
                            # Apply test indices if provided
                            if spec.test_indices:
                                try:
                                    ytv = [yt.inline[i] for i in spec.test_indices]
                                    ypv = [yp.inline[i] for i in spec.test_indices]
                                    from ..contracts.io import DataRef, DataRefType
                                    yt = DataRef(ref_type=DataRefType.INLINE, inline=ytv)
                                    yp = DataRef(ref_type=DataRefType.INLINE, inline=ypv)
                                except Exception:
                                    pass
                            # Prepare bindings
                            bindings = spec.bindings or {}
                            # Fallback to binding registry if present
                            if not bindings and self.binding_registry is not None:
                                try:
                                    bindings = {mid: self.binding_registry.get_metric_binding(mid) for mid in spec.metric_ids}
                                except Exception:
                                    bindings = {}
                            # Compute metrics
                            results: Dict[str, MetricComputeResult] = {}
                            for mid in spec.metric_ids:
                                b = bindings.get(mid)
                                if b is None:
                                    continue
                                mc_req = MetricComputeRequest(
                                    metric_id=mid,
                                    direction=self.metric_registry.get(mid).direction.value,
                                    y_true=yt,
                                    y_pred=yp,
                                    binding=b,
                                )
                                mres = self.metric_runner.compute(mc_req)
                                results[mid] = mres
                            res.eval_scores = {k: v.score for k, v in results.items()}
                    except Exception:
                        pass
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
