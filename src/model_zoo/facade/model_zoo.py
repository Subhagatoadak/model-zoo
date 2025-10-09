from __future__ import annotations
"""High‑level facade that orchestrates registries, runners, and evaluation."""
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
from ..utils.logging import get_logger
from ..events.bus import EventBus
import uuid

@dataclass
class ModelZoo:
    """Facade that wires together registries, validation, runners, and audit.

    Responsibilities:
      - Register models and metrics (pure metadata)
      - Validate parameters against hyperparameter schemas
      - Submit runs to the configured `ModelRunner`
      - Compute evaluation via `MetricRunner` and build R‑style summaries
    """
    model_registry: ModelRegistry
    metric_registry: MetricRegistry
    model_runner: ModelRunner
    metric_runner: MetricRunner
    validator: ParamValidationService
    audit: Optional[AuditLog] = None
    binding_registry: Optional[Any] = None  # optional external binding store
    _eval_specs: Dict[str, Dict[str, Any]] = None  # internal: run_id -> {spec, y_true_ref}
    _log = get_logger(__name__)
    _result_audits: Dict[str, str] = None  # run_id -> last audited status
    event_bus: Optional[EventBus] = None

    # registries
    def register_model(self, spec: ModelSpec, actor: str = "system") -> None:
        """Add a model spec to the registry and emit an audit event."""
        try:
            self.model_registry.register(spec)
            self._log.info("Registered model", extra={"model_id": spec.model_id})
        except Exception:
            self._log.exception("Failed to register model", extra={"model_id": spec.model_id})
            raise
        self._audit("REGISTER_MODEL", actor, spec.model_id, spec.to_dict())

    def register_metric(self, spec: EvalMetricSpec, actor: str = "system") -> None:
        """Add a metric spec to the registry and emit an audit event."""
        try:
            self.metric_registry.register(spec)
            self._log.info("Registered metric", extra={"metric_id": spec.metric_id})
        except Exception:
            self._log.exception("Failed to register metric", extra={"metric_id": spec.metric_id})
            raise
        self._audit("REGISTER_METRIC", actor, spec.metric_id, spec.to_dict())

    def list_models(self) -> List[ModelSpec]:
        """Return all registered models."""
        return self.model_registry.list()

    def list_metrics(self) -> List[EvalMetricSpec]:
        """Return all registered metrics."""
        return self.metric_registry.list()

    # runs
    def submit_run(self, req: ModelRunRequest, actor: str = "system") -> ModelRunHandle:
        """Validate params, audit submission, and delegate to the model runner."""
        spec = self.model_registry.get(req.model_id)
        self.validator.validate_model_params(spec, req.params)
        # If a binding registry exists, attach binding here (kept outside registries)
        if req.binding is None and self.binding_registry is not None:
            try:
                req.binding = self.binding_registry.get_model_binding(req.model_id)
            except Exception:
                req.binding = None
        self._audit("RUN_SUBMIT", actor, req.model_id, req.to_dict())
        trace_id = str(uuid.uuid4())
        self._log.info("Submitting run", extra={"model_id": req.model_id, "trace_id": trace_id})
        if self.event_bus:
            try:
                self.event_bus.publish("run.submitted", {"model_id": req.model_id, "trace_id": trace_id, "request": req.to_dict()})
            except Exception:
                self._log.debug("event_bus publish failed", exc_info=True)
        handle = self.model_runner.submit(req)
        # Initialize eval spec store
        if self._eval_specs is None:
            self._eval_specs = {}
        if req.evaluation is not None:
            y_key = req.evaluation.y_true_key
            y_ref = req.inputs.get(y_key)
            self._eval_specs[handle.run_id] = {"spec": req.evaluation, "y_true": y_ref, "trace_id": trace_id}
        else:
            self._eval_specs[handle.run_id] = {"spec": None, "y_true": None, "trace_id": trace_id}
        return handle

    def get_run_result(self, run_id: str, actor: str = "system") -> ModelRunResult:
        """Fetch current/final run result, compute evaluation, and build summary."""
        res = self.model_runner.result(run_id)
        # audit RUN_RESULT only on status change
        if self._result_audits is None:
            self._result_audits = {}
        last = self._result_audits.get(run_id)
        if last != res.status.value:
            self._audit("RUN_RESULT", actor, run_id, {"status": res.status.value})
            self._result_audits[run_id] = res.status.value
            if self.event_bus:
                try:
                    self.event_bus.publish("run.status_changed", {"run_id": run_id, "status": res.status.value})
                except Exception:
                    self._log.debug("event_bus publish failed", exc_info=True)
        # attach trace_id to metadata
        try:
            plan = self._eval_specs.get(run_id) if getattr(self, "_eval_specs", None) else None
            if plan:
                trace_id = plan.get("trace_id")
                if trace_id:
                    if isinstance(res.metadata, dict):
                        res.metadata.setdefault("trace_id", trace_id)
                    else:
                        res.metadata = {"trace_id": trace_id}
        except Exception:
            self._log.debug("Failed to attach trace_id", exc_info=True)
        # Inline evaluation: compute train/valid metrics and R-style summary
        if getattr(self, "_eval_specs", None) and run_id in self._eval_specs:
            plan = self._eval_specs[run_id]
            spec: EvaluationSpec = plan["spec"]
            y_true_ref = plan.get("y_true")
            if res.status.name == "SUCCEEDED" and res.predictions is not None and y_true_ref is not None:
                try:
                    if y_true_ref.ref_type.name == "INLINE" and res.predictions.ref_type.name == "INLINE":
                        from ..contracts.io import DataRef, DataRefType

                        def slice_ref(r, idxs):
                            if not idxs:
                                return r
                            vals = [r.inline[i] for i in idxs]
                            return DataRef(ref_type=DataRefType.INLINE, inline=vals)

                        # Prepare metric bindings
                        bindings = spec.bindings or {}
                        if not bindings and self.binding_registry is not None:
                            try:
                                bindings = {mid: self.binding_registry.get_metric_binding(mid) for mid in spec.metric_ids}
                            except Exception:
                                bindings = {}

                        # Compute train metrics
                        if spec.compute_train:
                            yt_train = slice_ref(y_true_ref, spec.train_indices)
                            yp_train = slice_ref(res.predictions, spec.train_indices)
                            res.eval_train = self._compute_metrics_set(spec.metric_ids, yt_train, yp_train, bindings)

                        # Compute validation metrics
                        if spec.compute_valid and spec.test_indices:
                            yt_test = slice_ref(y_true_ref, spec.test_indices)
                            yp_test = slice_ref(res.predictions, spec.test_indices)
                            res.eval_valid = self._compute_metrics_set(spec.metric_ids, yt_test, yp_test, bindings)
                except Exception:
                    self._log.exception("Evaluation failed", extra={"run_id": run_id})
                finally:
                    # Always attempt to build a summary, even if metrics failed
                    try:
                        res.summary = self._build_r_style_summary(run_id, res, spec, y_true_ref)
                        if self.event_bus:
                            try:
                                self.event_bus.publish("run.evaluated", {"run_id": run_id, "metrics": {"train": res.eval_train, "validation": res.eval_valid}})
                            except Exception:
                                self._log.debug("event_bus publish failed", exc_info=True)
                    except Exception:
                        self._log.debug("Failed to build summary", exc_info=True)
        # If summary is still empty (e.g., no evaluation spec provided), build a minimal one
        if res.status.name == "SUCCEEDED" and not res.summary:
            try:
                plan = self._eval_specs.get(run_id) if getattr(self, "_eval_specs", None) else None
                y_true_ref = plan.get("y_true") if plan else None
                res.summary = self._build_r_style_summary(run_id, res, None, y_true_ref)
            except Exception:
                self._log.debug("Failed to build minimal summary", exc_info=True)
        return res

    def _compute_metrics_set(self, metric_ids: List[str], y_true, y_pred, bindings: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for mid in metric_ids:
            b = bindings.get(mid)
            if b is None:
                continue
            # Resolve direction if metric is registered; otherwise default to 'higher_is_better'
            try:
                direction = self.metric_registry.get(mid).direction.value
            except Exception:
                self._log.warning("Metric id not registered; using default direction", extra={"metric_id": mid})
                direction = "higher_is_better"
            try:
                mc_req = MetricComputeRequest(
                    metric_id=mid,
                    direction=direction,
                    y_true=y_true,
                    y_pred=y_pred,
                    binding=b,
                )
                mres = self.metric_runner.compute(mc_req)
                out[mid] = mres.score
            except Exception:
                self._log.exception("Metric compute failed", extra={"metric_id": mid})
                # Continue to next metric
        return out

    def _build_r_style_summary(self, run_id: str, res: ModelRunResult, spec: Optional[EvaluationSpec], y_true_ref: Any) -> Dict[str, Any]:
        # Minimal R-like summary with sections: Call, Residuals summary, Coefficients, Fit stats, Metrics
        # Prefer explicit call info if runner provided; fall back to binding
        call = {}
        try:
            meta = res.metadata if isinstance(res.metadata, dict) else {}
            call = meta.get("call") or meta.get("binding", {})
        except Exception:
            call = {}

        # Coefficients: rely on diagnostics for coefficient-like entries
        coefficients = res.diagnostics if isinstance(res.diagnostics, dict) else {}

        # Fit stats (train) if available
        fit: Dict[str, Any] = {}
        if res.eval_train:
            # Common keys: r2_v1, mse_v1
            r2 = res.eval_train.get("r2_v1")
            mse = res.eval_train.get("mse_v1")
            fit.update({k: v for k, v in [("r_squared", r2), ("mse", mse)] if v is not None})

        sizes = {}
        try:
            n = len(y_true_ref.inline) if (y_true_ref is not None and getattr(y_true_ref, "inline", None) is not None) else None
            n_train = len(spec.train_indices) if (spec is not None and spec.train_indices) else None
            n_valid = len(spec.test_indices) if (spec is not None and spec.test_indices) else None
            sizes = {"n": n, "n_train": n_train, "n_valid": n_valid}
        except Exception:
            pass

        summary = {
            "Call": call,
            "Coefficients": coefficients,
            "Fit": fit,
            "Sizes": sizes,
            "Metrics": {
                "train": res.eval_train,
                "validation": res.eval_valid,
            },
        }
        return summary

    def compute_metric(self, req: MetricComputeRequest, actor: str = "system") -> MetricComputeResult:
        """Validate params, audit, and delegate to the metric runner."""
        spec = self.metric_registry.get(req.metric_id)
        self.validator.validate_metric_params(spec, req.params)
        self._audit("METRIC_COMPUTE", actor, req.metric_id, req.to_dict())
        try:
            return self.metric_runner.compute(req)
        except Exception:
            self._log.exception("Metric compute failed", extra={"metric_id": req.metric_id})
            raise

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
                self._log.error("Unknown metric_id in compute_all_metrics", extra={"metric_id": mid})
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
                self._log.error("Missing binding for metric", extra={"metric_id": mid})
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
            try:
                results[mid] = self.compute_metric(req, actor=actor)
            except Exception:
                self._log.exception("Metric compute failed in batch", extra={"metric_id": mid})
                raise
        return results

    def _audit(self, kind: str, who: str, ref_id: str, payload: Dict[str, Any]) -> None:
        if not self.audit: return
        from datetime import datetime, timezone
        evt = AuditEvent(kind=kind, who=who, when=datetime.now(timezone.utc).isoformat(), ref_id=ref_id, payload=payload)
        self.audit.append(evt)
