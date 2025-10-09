from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from threading import Lock
from typing import Any, Callable, Dict, Optional

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
from ..utils.logging import get_logger


def _default_load(ref: DataRef) -> Any:
    """Resolve a DataRef to a Python object.

    Default behavior:
    - INLINE: return the inline payload as-is
    - URI: return the DataRef itself (caller decides how to load)
    """
    if ref.ref_type == DataRefType.INLINE:
        return ref.inline
    return ref  # pass-through for URI by default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class _Job:
    req: ModelRunRequest
    status: RunStatus
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    result: Optional[ModelRunResult] = None
    future: Optional[Future] = None
    cancelled: bool = False


class LibraryModelRunner(ModelRunner):
    """In-process runner that invokes Python callables via an ExecutionBinding.

    Expected callable signature (recommended):
        def run(inputs: dict, params: dict, context: ExecutionContext) -> dict | ModelRunResult

    The callable may return either:
      - A ModelRunResult instance
      - A dictionary with any of the keys: predictions, diagnostics, artifacts, metadata
        where predictions and artifact values may be raw objects. They will be wrapped
        into DataRef(INLINE) automatically.
    """

    def __init__(
        self,
        max_workers: int = 4,
        data_loader: Optional[Callable[[DataRef], Any]] = None,
    ) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mz-lib-runner")
        self._jobs: Dict[str, _Job] = {}
        self._lock = Lock()
        self._load = data_loader or _default_load
        self._log = get_logger(__name__)

    def submit(self, req: ModelRunRequest) -> ModelRunHandle:
        binding = self._require_library_binding(req.binding)
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        self._log.info("Submitting library run", extra={"run_id": run_id, "bind_module": binding.module, "bind_callable": binding.callable})
        job = _Job(req=req, status=RunStatus.STARTED, started_at=_now_iso())
        with self._lock:
            self._jobs[run_id] = job

        future = self._executor.submit(self._execute, run_id, req, binding)
        job.future = future
        return ModelRunHandle(run_id=run_id, status=RunStatus.STARTED)

    def status(self, run_id: str) -> RunStatus:
        job = self._jobs.get(run_id)
        if not job:
            raise KeyError(f"Unknown run_id: {run_id}")
        return job.status

    def result(self, run_id: str) -> ModelRunResult:
        job = self._jobs.get(run_id)
        if not job:
            raise KeyError(f"Unknown run_id: {run_id}")
        # If execution finished, return final result
        if job.result is not None:
            return job.result
        # If still running, return a partial placeholder
        return ModelRunResult(
            run_id=run_id,
            status=job.status,
            metadata={"in_progress": True},
            started_at=job.started_at,
            ended_at=job.ended_at,
        )

    def cancel(self, run_id: str) -> None:
        job = self._jobs.get(run_id)
        if not job:
            return
        job.cancelled = True
        # Best-effort: cannot force-stop threads; mark as cancelled if not finished
        if job.status in (RunStatus.QUEUED, RunStatus.STARTED):
            job.status = RunStatus.CANCELLED
            job.ended_at = _now_iso()

    # Internal helpers
    def _execute(self, run_id: str, req: ModelRunRequest, binding: ExecutionBinding) -> None:
        job = self._jobs[run_id]
        try:
            func = self._resolve_callable(binding)
            loaded_inputs = {k: self._load(v) for k, v in (req.inputs or {}).items()}
            # Recommended callable signature, but we are permissive
            out = func(loaded_inputs, req.params, req.context)

            if isinstance(out, ModelRunResult):
                result = out
            else:
                # Normalize dict-like outputs
                if not isinstance(out, dict):
                    out = {"predictions": out}
                preds = out.get("predictions")
                artifacts = out.get("artifacts", {})
                diagnostics = out.get("diagnostics", {})
                metadata = out.get("metadata", {})

                def wrap(val):
                    if isinstance(val, DataRef):
                        return val
                    return DataRef(ref_type=DataRefType.INLINE, inline=val)

                result = ModelRunResult(
                    run_id=run_id,
                    status=RunStatus.SUCCEEDED,
                    predictions=None if preds is None else wrap(preds),
                    diagnostics=diagnostics,
                    artifacts={k: wrap(v) for k, v in artifacts.items()},
                    metadata=metadata,
                    started_at=job.started_at,
                    ended_at=_now_iso(),
                )

            # Enrich metadata with binding + call info and params
            try:
                meta_extra = {
                    "binding": binding.to_dict(),
                    "call": {
                        "module": binding.module,
                        "callable": binding.callable,
                        "params": req.params,
                    },
                }
                if isinstance(result.metadata, dict):
                    result.metadata.update(meta_extra)
                else:
                    result.metadata = meta_extra
            except Exception:
                pass

            # Success path
            job.status = result.status
            job.result = result
            if not job.ended_at:
                job.ended_at = _now_iso()
            self._log.info("Library run finished", extra={"run_id": run_id, "status": job.status.value})
        except Exception as e:
            job.status = RunStatus.FAILED if not job.cancelled else RunStatus.CANCELLED
            job.result = ModelRunResult(
                run_id=run_id,
                status=job.status,
                predictions=None,
                diagnostics={"error": str(e)},
                metadata={"binding": binding.to_dict()},
                started_at=job.started_at,
                ended_at=_now_iso(),
            )
            self._log.exception("Library run failed", extra={"run_id": run_id})

    def _require_library_binding(self, binding: Optional[ExecutionBinding]) -> ExecutionBinding:
        if binding is None:
            raise ValueError("ModelRunRequest.binding is required for LibraryModelRunner")
        if binding.kind != BindingKind.LIBRARY:
            raise ValueError(f"Binding kind must be 'library', got {binding.kind.value}")
        if not binding.module or not binding.callable:
            raise ValueError("Binding must specify 'module' and 'callable'")
        return binding

    def _resolve_callable(self, binding: ExecutionBinding) -> Callable[..., Any]:
        mod = import_module(binding.module)
        func = getattr(mod, binding.callable)
        if not callable(func):
            raise TypeError(f"Resolved object '{binding.module}.{binding.callable}' is not callable")
        return func


class LibraryMetricRunner(MetricRunner):
    """In-process metric runner that invokes Python callables.

    Expected callable signature (recommended):
        def compute(y_true, y_pred=None, proba=None, samples=None, params: dict = None, extra: dict = None) -> float | dict

    The callable may return either a numeric score, or a dict with keys
    'score', 'artifacts', and 'metadata'.
    """

    def __init__(self, data_loader: Optional[Callable[[DataRef], Any]] = None) -> None:
        self._load = data_loader or _default_load

    def compute(self, req: MetricComputeRequest) -> MetricComputeResult:
        binding = self._require_library_binding(req.binding)
        func = self._resolve_callable(binding)

        enc = lambda r: None if r is None else self._load(r)
        out = func(
            enc(req.y_true),
            y_pred=enc(req.y_pred),
            proba=enc(req.proba),
            samples=enc(req.samples),
            params=req.params,
            extra=req.extra,
        )

        if isinstance(out, MetricComputeResult):
            return out

        if isinstance(out, dict):
            score = out.get("score")
            artifacts = out.get("artifacts", {})
            metadata = out.get("metadata", {})
        else:
            score = float(out) if out is not None else None
            artifacts, metadata = {}, {}

        def wrap(val):
            if isinstance(val, DataRef):
                return val
            return DataRef(ref_type=DataRefType.INLINE, inline=val)

        return MetricComputeResult(
            score=score,
            artifacts={k: wrap(v) for k, v in artifacts.items()},
            metadata=metadata,
        )

    def _require_library_binding(self, binding: Optional[ExecutionBinding]) -> ExecutionBinding:
        if binding is None:
            raise ValueError("MetricComputeRequest.binding is required for LibraryMetricRunner")
        if binding.kind != BindingKind.LIBRARY:
            raise ValueError(f"Binding kind must be 'library', got {binding.kind.value}")
        if not binding.module or not binding.callable:
            raise ValueError("Binding must specify 'module' and 'callable'")
        return binding

    def _resolve_callable(self, binding: ExecutionBinding) -> Callable[..., Any]:
        mod = import_module(binding.module)
        func = getattr(mod, binding.callable)
        if not callable(func):
            raise TypeError(f"Resolved object '{binding.module}.{binding.callable}' is not callable")
        return func
