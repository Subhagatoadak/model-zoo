"""Run store interfaces and an in-memory implementation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
from ..contracts.run_requests import ModelRunResult, RunStatus
from threading import Lock
from ..contracts.io import DataRef, DataRefType

try:
    import psycopg
    from psycopg import sql
    from psycopg.types.json import Json
except ImportError:  # pragma: no cover - optional dependency
    psycopg = None  # type: ignore
    sql = None  # type: ignore
    Json = None  # type: ignore


@dataclass
class RunRecord:
    """A stored run record containing the latest `ModelRunResult`."""
    run_id: str
    result: ModelRunResult


class RunStore:
    """Abstract interface for persisting run records."""

    def save(self, rec: RunRecord) -> None:
        """Persist or update a run record."""
        raise NotImplementedError

    def get(self, run_id: str) -> Optional[RunRecord]:
        """Retrieve a run record by id, or None if absent."""
        raise NotImplementedError


class InMemoryRunStore(RunStore):
    def __init__(self) -> None:
        self._recs: Dict[str, RunRecord] = {}
        self._lock = Lock()

    def save(self, rec: RunRecord) -> None:
        """Store the run record in memory, replacing any existing one."""
        with self._lock:
            self._recs[rec.run_id] = rec

    def get(self, run_id: str) -> Optional[RunRecord]:
        """Return a previously saved RunRecord, if present."""
        with self._lock:
            return self._recs.get(run_id)


def _dataref_from_payload(payload: Optional[Dict[str, Any]]) -> Optional[DataRef]:
    if payload is None:
        return None
    ref_type = payload.get("ref_type")
    if not ref_type:
        return None
    return DataRef(ref_type=DataRefType(ref_type), uri=payload.get("uri"), inline=payload.get("inline"), format=payload.get("format"), schema_hint=payload.get("schema_hint") or {})


def _dataref_to_payload(ref: Optional[DataRef]) -> Optional[Dict[str, Any]]:
    if ref is None:
        return None
    return ref.to_dict()


def _result_to_payload(result: ModelRunResult) -> Dict[str, Any]:
    return {
        "run_id": result.run_id,
        "status": result.status.value,
        "predictions": _dataref_to_payload(result.predictions),
        "diagnostics": result.diagnostics,
        "artifacts": {k: _dataref_to_payload(v) for k, v in (result.artifacts or {}).items()},
        "metadata": result.metadata,
        "started_at": result.started_at,
        "ended_at": result.ended_at,
        "eval_train": result.eval_train,
        "eval_valid": result.eval_valid,
        "summary": result.summary,
    }


def _payload_to_result(payload: Dict[str, Any]) -> ModelRunResult:
    return ModelRunResult(
        run_id=payload.get("run_id"),
        status=RunStatus(payload.get("status")),
        predictions=_dataref_from_payload(payload.get("predictions")),
        diagnostics=payload.get("diagnostics") or {},
        artifacts={k: _dataref_from_payload(v) for k, v in (payload.get("artifacts") or {}).items() if v is not None},
        metadata=payload.get("metadata") or {},
        started_at=payload.get("started_at"),
        ended_at=payload.get("ended_at"),
        eval_train=payload.get("eval_train") or {},
        eval_valid=payload.get("eval_valid") or {},
        summary=payload.get("summary") or {},
    )


class PostgresRunStore(RunStore):
    """Persist run records in PostgreSQL using psycopg."""

    def __init__(self, dsn: str, table_name: str = "model_runs", connect_timeout: int = 10) -> None:
        if psycopg is None:
            raise ImportError("psycopg is required for PostgresRunStore. Install the 'psycopg' package.")
        self._dsn = dsn
        self._table_name = table_name
        self._connect_kwargs = {"connect_timeout": connect_timeout}
        self._lock = Lock()
        self._ensure_table()

    def _get_conn(self):
        return psycopg.connect(self._dsn, **self._connect_kwargs)

    def _ensure_table(self) -> None:
        with self._get_conn() as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {table} (
                            run_id TEXT PRIMARY KEY,
                            status TEXT NOT NULL,
                            payload JSONB NOT NULL,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            updated_at TIMESTAMPTZ DEFAULT NOW()
                        )
                        """
                    ).format(table=sql.Identifier(self._table_name))
                )

    def save(self, rec: RunRecord) -> None:
        payload = _result_to_payload(rec.result)
        with self._lock:
            with self._get_conn() as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL(
                            """
                            INSERT INTO {table} (run_id, status, payload, updated_at)
                            VALUES (%s, %s, %s, NOW())
                            ON CONFLICT (run_id) DO UPDATE
                            SET status = EXCLUDED.status,
                                payload = EXCLUDED.payload,
                                updated_at = NOW()
                            """
                        ).format(table=sql.Identifier(self._table_name)),
                        (rec.run_id, rec.result.status.value, Json(payload)),
                    )

    def get(self, run_id: str) -> Optional[RunRecord]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT payload FROM {table} WHERE run_id = %s").format(table=sql.Identifier(self._table_name)),
                    (run_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                payload = row[0]
                if not isinstance(payload, dict):
                    return None
                result = _payload_to_result(payload)
                return RunRecord(run_id=run_id, result=result)
