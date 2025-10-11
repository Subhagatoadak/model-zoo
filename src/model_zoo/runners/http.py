"""HTTP-based runners (submit/compute via API endpoints).

These are conventional implementations: submit via POST, then poll or fetch
results via GET endpoints. Integrate with a BindingRegistry to manage
endpoints per model/metric.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .base import ModelRunner, MetricRunner
from ..bindings.base import ExecutionBinding, BindingKind
from ..contracts.io import DataRef, DataRefType
from ..contracts.run_requests import (
    ModelRunRequest,
    ModelRunHandle,
    ModelRunResult,
    MetricComputeRequest,
    MetricComputeResult,
    RunStatus,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_dataref(val: Any) -> DataRef:
    if isinstance(val, DataRef):
        return val
    return DataRef(ref_type=DataRefType.INLINE, inline=val)


class HttpModelRunner(ModelRunner):
    """HTTP-based runner that calls an API endpoint described by an ExecutionBinding.

    Expected API semantics (conventional):
      - submit:   POST to binding.endpoint with the request payload -> {run_id, status}
      - status:   GET  binding.endpoint + "/{run_id}/status" -> {status}
      - result:   GET  binding.endpoint + "/{run_id}/result" -> {status, predictions, artifacts, diagnostics, metadata}
      - cancel:   POST binding.endpoint + "/{run_id}/cancel" -> {ok: true}
    """

    def __init__(self, http_client: Optional[Any] = None) -> None:
        """Create a runner with an optional requests-like client session."""
        self._http = http_client

    def submit(self, req: ModelRunRequest) -> ModelRunHandle:
        """POST the run request to `binding.endpoint` and return a handle."""
        binding = self._require_http_binding(req.binding)
        session = self._session()
        resp = session.post(binding.endpoint, json=req.to_dict())
        resp.raise_for_status()
        data = resp.json()
        return ModelRunHandle(run_id=data["run_id"], status=RunStatus(data.get("status", RunStatus.STARTED.value)))

    def status(self, run_id: str) -> RunStatus:
        raise NotImplementedError("Use result() or implement a status endpoint in your API.")

    def result(self, run_id: str) -> ModelRunResult:
        # Without a stored endpoint per run_id, this method requires the caller to include
        # the endpoint in a run-specific binding registry. In many deployments, ModelZoo
        # will manage this layer. For portability, we expose a single-call fallback on submit().
        raise NotImplementedError("HttpModelRunner.result requires a binding registry to resolve endpoints per run.")

    def cancel(self, run_id: str) -> None:
        raise NotImplementedError("HttpModelRunner.cancel requires a binding registry to resolve endpoints per run.")

    def _require_http_binding(self, binding: Optional[ExecutionBinding]) -> ExecutionBinding:
        if binding is None:
            raise ValueError("ModelRunRequest.binding is required for HttpModelRunner")
        if binding.kind != BindingKind.API:
            raise ValueError(f"Binding kind must be 'api', got {binding.kind.value}")
        if not binding.endpoint:
            raise ValueError("Binding must specify 'endpoint'")
        return binding

    def _session(self):
        """Return an HTTP session (requests.Session)."""
        if self._http is not None:
            return self._http
        try:
            import requests  # lazy import
        except Exception as e:
            raise RuntimeError("The 'requests' package is required for HttpModelRunner") from e
        return requests.Session()


class HttpMetricRunner(MetricRunner):
    """HTTP-based metric runner.

    Expected API semantics (conventional):
      - compute: POST to binding.endpoint with the metric request payload -> {score, artifacts, metadata}
    """

    def __init__(self, http_client: Optional[Any] = None) -> None:
        """Create a metric runner with an optional requests-like session."""
        self._http = http_client

    def compute(self, req: MetricComputeRequest) -> MetricComputeResult:
        """POST the metric request and parse a score/artifacts/metadata response."""
        binding = self._require_http_binding(req.binding)
        session = self._session()
        resp = session.post(binding.endpoint, json=req.to_dict())
        resp.raise_for_status()
        data: Dict[str, Any] = resp.json()
        score = data.get("score")
        artifacts = data.get("artifacts", {})
        metadata = data.get("metadata", {})
        return MetricComputeResult(
            score=score,
            artifacts={k: _as_dataref(v) for k, v in artifacts.items()},
            metadata=metadata,
        )

    def _require_http_binding(self, binding: Optional[ExecutionBinding]) -> ExecutionBinding:
        if binding is None:
            raise ValueError("MetricComputeRequest.binding is required for HttpMetricRunner")
        if binding.kind != BindingKind.API:
            raise ValueError(f"Binding kind must be 'api', got {binding.kind.value}")
        if not binding.endpoint:
            raise ValueError("Binding must specify 'endpoint'")
        return binding

    def _session(self):
        """Return an HTTP session (requests.Session)."""
        if self._http is not None:
            return self._http
        try:
            import requests
        except Exception as e:
            raise RuntimeError("The 'requests' package is required for HttpMetricRunner") from e
        return requests.Session()
