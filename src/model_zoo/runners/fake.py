from __future__ import annotations
from datetime import datetime, timezone
from ..contracts.run_requests import (
    ModelRunRequest, ModelRunHandle, ModelRunResult, RunStatus,
    MetricComputeRequest, MetricComputeResult
)
from ..contracts.io import DataRef, DataRefType

class FakeModelRunner:
    def __init__(self) -> None:
        self._store = {}

    def submit(self, req: ModelRunRequest) -> ModelRunHandle:
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        # Immediately "finish" for demo
        self._store[run_id] = ModelRunResult(
            run_id=run_id, status=RunStatus.SUCCEEDED,
            predictions=DataRef(ref_type=DataRefType.URI, uri="s3://demo/preds.parquet", format="parquet"),
            metadata={"binding": None if req.binding is None else req.binding.to_dict()},
            started_at=datetime.now(timezone.utc).isoformat(),
            ended_at=datetime.now(timezone.utc).isoformat(),
        )
        return ModelRunHandle(run_id=run_id, status=RunStatus.STARTED)

    def status(self, run_id: str) -> RunStatus:
        return RunStatus.SUCCEEDED if run_id in self._store else RunStatus.FAILED

    def result(self, run_id: str) -> ModelRunResult:
        return self._store[run_id]

    def cancel(self, run_id: str) -> None:
        return

class FakeMetricRunner:
    def compute(self, req: MetricComputeRequest) -> MetricComputeResult:
        return MetricComputeResult(score=0.0, metadata={"demo": True})
