"""FastAPI application exposing endpoints to register models and configurations."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

import psycopg
from psycopg.types.json import Json

from ..facade.model_zoo import ModelZoo
from ..registry.models import ModelRegistry, ModelSpec, ModelKind
from ..registry.metrics import MetricRegistry, EvalMetricSpec, MetricKind, MetricDirection
from ..contracts.validation import ParamValidationService
from ..contracts.validators import DefaultParamValidator
from ..contracts.run_requests import EvaluationSpec, ModelRunRequest, RunStatus
from ..contracts.io import DataRef, DataRefType, ExecutionContext
from ..stores.run_store import InMemoryRunStore
from ..runners.library import LibraryModelRunner, LibraryMetricRunner
from ..utils.config_loader import (
    load_model_configuration_from_dict,
    ModelConfiguration,
)
from ..utils.variable_processing import build_variables
from ..utils.env import read_env_file, build_postgres_dsn
from ..bindings.registry import InMemoryBindingRegistry
from ..bindings.base import ExecutionBinding, BindingKind
from ..utils.summary import r_style_summary_string

import time
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from threading import Lock
from concurrent.futures import ThreadPoolExecutor


class RegistryDatabase:
    """Lightweight helper around psycopg to persist registry artifacts."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        with psycopg.connect(self._dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS model_configs (
                        model_id TEXT NOT NULL,
                        environment TEXT,
                        config JSONB NOT NULL,
                        registered_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (model_id, environment)
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS model_specs (
                        model_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        version TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        hyperparams JSONB NOT NULL,
                        tags JSONB DEFAULT '[]'::jsonb,
                        metadata JSONB DEFAULT '{}'::jsonb,
                        registered_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )

    def upsert_config(self, model_id: str, environment: Optional[str], config: Dict[str, Any]) -> None:
        with psycopg.connect(self._dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_configs (model_id, environment, config, updated_at)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (model_id, environment) DO UPDATE
                    SET config = EXCLUDED.config,
                        updated_at = NOW()
                    """,
                    (model_id, environment, Json(config)),
                )

    def upsert_spec(self, spec: ModelSpec) -> None:
        with psycopg.connect(self._dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_specs (model_id, name, version, kind, hyperparams, tags, metadata, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (model_id) DO UPDATE
                    SET name = EXCLUDED.name,
                        version = EXCLUDED.version,
                        kind = EXCLUDED.kind,
                        hyperparams = EXCLUDED.hyperparams,
                        tags = EXCLUDED.tags,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    """,
                    (
                        spec.model_id,
                        spec.name,
                        spec.version,
                        spec.kind.value,
                        Json(spec.hyperparams.to_json_schema()),
                        Json(spec.tags),
                        Json(spec.metadata),
                    ),
                )

    def list_specs(self) -> List[Dict[str, Any]]:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT model_id, name, version, kind, hyperparams, tags, metadata, registered_at, updated_at FROM model_specs ORDER BY model_id"
                )
                rows = cur.fetchall()
        specs = []
        for row in rows:
            specs.append(
                {
                    "model_id": row[0],
                    "name": row[1],
                    "version": row[2],
                    "kind": row[3],
                    "hyperparams": row[4],
                    "tags": row[5],
                    "metadata": row[6],
                    "registered_at": row[7].isoformat() if row[7] else None,
                    "updated_at": row[8].isoformat() if row[8] else None,
                }
            )
        return specs

    def fetch_spec(self, model_id: str) -> Optional[Dict[str, Any]]:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT model_id, name, version, kind, hyperparams, tags, metadata, registered_at, updated_at FROM model_specs WHERE model_id = %s",
                    (model_id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return {
            "model_id": row[0],
            "name": row[1],
            "version": row[2],
            "kind": row[3],
            "hyperparams": row[4],
            "tags": row[5],
            "metadata": row[6],
            "registered_at": row[7].isoformat() if row[7] else None,
            "updated_at": row[8].isoformat() if row[8] else None,
        }

    def list_configs(self, model_id: str, environment: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT environment, config, registered_at, updated_at FROM model_configs WHERE model_id = %s"
        params: List[Any] = [model_id]
        if environment is not None:
            query += " AND (environment = %s OR (environment IS NULL AND %s IS NULL))"
            params.extend([environment, environment])
        query += " ORDER BY environment NULLS LAST"
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        return [
            {
                "environment": row[0],
                "config": row[1],
                "registered_at": row[2].isoformat() if row[2] else None,
                "updated_at": row[3].isoformat() if row[3] else None,
            }
            for row in rows
        ]

    def get_config(self, model_id: str, environment: Optional[str]) -> Optional[Dict[str, Any]]:
        query = "SELECT config FROM model_configs WHERE model_id = %s"
        params: List[Any] = [model_id]
        if environment is not None:
            query += " AND environment = %s"
            params.append(environment)
        else:
            query += " AND environment IS NULL"
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                row = cur.fetchone()
                if row:
                    return row[0]
        if environment is not None:
            # fallback to environment-agnostic config
            with psycopg.connect(self._dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT config FROM model_configs WHERE model_id = %s AND environment IS NULL",
                        (model_id,),
                    )
                    row = cur.fetchone()
                    if row:
                        return row[0]
        return None


def _resolve_postgres_dsn() -> str:
    env_file = os.environ.get("MODEL_ZOO_ENV_FILE", ".env")
    env_settings: Dict[str, str] = {}
    if Path(env_file).exists():
        try:
            env_settings = read_env_file(env_file)
        except FileNotFoundError:
            env_settings = {}

    dsn = os.environ.get("POSTGRES_DSN") or env_settings.get("POSTGRES_DSN")
    if dsn:
        return dsn

    merged: Dict[str, str] = {}
    merged.update(env_settings)
    merged.update({k: v for k, v in os.environ.items() if k.startswith("POSTGRES_")})
    try:
        dsn, _ = build_postgres_dsn(merged)
    except ValueError as exc:
        raise RuntimeError(
            "POSTGRES_DSN or POSTGRES_* credentials must be provided via environment or .env"
        ) from exc
    return dsn


postgres_dsn = _resolve_postgres_dsn()
database = RegistryDatabase(postgres_dsn)
model_registry = ModelRegistry()
metric_registry = MetricRegistry()
binding_registry = InMemoryBindingRegistry()
validator: ParamValidationService = DefaultParamValidator()
run_manager = RunManager()
run_executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI(title="Model Zoo Registry API", version="0.1.0")


class ConfigRegistrationRequest(BaseModel):
    env: Optional[str] = Field(default=None, description="Environment label for this configuration")
    model_id: Optional[str] = Field(default=None, description="Optional override for model id")
    config: Dict[str, Any]


class ConfigRegistrationResponse(BaseModel):
    model_id: str
    environment: Optional[str]
    spec: Dict[str, Any]
    stored: bool = True


class ModelSpecRequest(BaseModel):
    model_id: str
    name: str
    version: str
    kind: str = Field(description="One of: library, api, pipeline")
    hyperparameters: List[Dict[str, Any]] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelSpecResponse(BaseModel):
    model_id: str
    name: str
    version: str
    kind: str
    hyperparams: Dict[str, Any]
    tags: List[str]
    metadata: Dict[str, Any]


class RunEvaluationRequest(BaseModel):
    metric_ids: List[str]
    y_true_key: str = "y"
    compute_train: bool = True
    compute_valid: bool = True
    train_indices: Optional[List[int]] = None
    test_indices: Optional[List[int]] = None


class RunRequest(BaseModel):
    model_id: str
    env: Optional[str] = Field(default=None, description="Environment label for selecting configuration")
    data_sources: Dict[str, Any]
    params: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    evaluation: Optional[RunEvaluationRequest] = None
    inputs: Optional[List[str]] = Field(default=None, description="Optional explicit list of input keys")


class RunResponse(BaseModel):
    run_id: str
    status: str
    predictions: Any = None
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    eval_train: Dict[str, float] = Field(default_factory=dict)
    eval_valid: Dict[str, float] = Field(default_factory=dict)
    summary: Dict[str, Any] = Field(default_factory=dict)
    summary_text: Optional[str] = None


class RunSubmissionResponse(BaseModel):
    run_id: str
    status: str


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    submitted_at: str
    completed_at: Optional[str] = None
    result: Optional[RunResponse] = None
    error: Optional[str] = None


def _register_model_spec_from_configuration(cfg: ModelConfiguration) -> ModelSpec:
    model_info = cfg.model
    spec = ModelSpec(
        model_id=model_info["id"],
        name=model_info.get("name", model_info["id"]),
        version=model_info.get("version", "0.0.1"),
        kind=ModelKind[model_info.get("kind", "library").upper()],
        hyperparams=cfg.hyperparam_space,
        tags=model_info.get("tags", []),
        metadata=model_info.get("metadata", {}),
    )
    try:
        model_registry.register(spec)
    except KeyError:
        model_registry.replace(spec.model_id, spec)
    database.upsert_spec(spec)

    binding_info = model_info.get("binding") or {}
    if binding_info:
        binding = ExecutionBinding(
            bound_id=spec.model_id,
            kind=BindingKind.LIBRARY if (binding_info.get("kind") or "library").lower() == "library" else BindingKind.API,
            module=binding_info["module"],
            callable=binding_info["callable"],
        )
        binding_registry.register_model_binding(spec.model_id, binding)

    metrics_cfg = cfg.raw.get("metrics") or []
    for metric in metrics_cfg:
        try:
            kind = MetricKind[metric.get("kind", "library").upper()]
        except KeyError:
            kind = MetricKind.LIBRARY
        direction_value = metric.get("direction", "higher_is_better").upper()
        try:
            direction = MetricDirection[direction_value]
        except KeyError:
            direction = MetricDirection.HIGHER_IS_BETTER
        metric_spec = EvalMetricSpec(
            metric_id=metric["metric_id"],
            name=metric.get("name", metric["metric_id"]),
            version=metric.get("version", "0.0.1"),
            direction=direction,
            kind=kind,
            tags=metric.get("tags", []),
        )
        try:
            metric_registry.register(metric_spec)
        except KeyError:
            metric_registry.replace(metric_spec.metric_id, metric_spec)
        binding = metric.get("binding") or {}
        if binding:
            metric_binding = ExecutionBinding(
                bound_id=metric_spec.metric_id,
                kind=BindingKind.LIBRARY if (binding.get("kind") or "library").lower() == "library" else BindingKind.API,
                module=binding["module"],
                callable=binding["callable"],
            )
            binding_registry.register_metric_binding(metric_spec.metric_id, metric_binding)
    return spec


def _register_simple_model_spec(payload: ModelSpecRequest) -> ModelSpec:
    try:
        kind = ModelKind[payload.kind.upper()]
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown model kind: {payload.kind}") from exc
    from ..utils.config_tables import hyperparam_space_from_master

    try:
        hyperparams = hyperparam_space_from_master(payload.hyperparameters)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid hyperparameter schema: {exc}") from exc

    spec = ModelSpec(
        model_id=payload.model_id,
        name=payload.name,
        version=payload.version,
        kind=kind,
        hyperparams=hyperparams,
        tags=payload.tags,
        metadata=payload.metadata,
    )
    try:
        model_registry.register(spec)
    except KeyError:
        model_registry.replace(spec.model_id, spec)
    database.upsert_spec(spec)
    return spec


def get_database() -> RegistryDatabase:
    return database


def _dataref_payload(ref: Optional[DataRef]) -> Any:
    if ref is None:
        return None
    if ref.ref_type == DataRefType.INLINE:
        return ref.inline
    return ref.to_dict()


def _artifacts_payload(artifacts: Dict[str, DataRef]) -> Dict[str, Any]:
    return {name: _dataref_payload(ref) for name, ref in artifacts.items()}


def _prepare_evaluation(
    cfg: ModelConfiguration,
    request_eval: Optional[RunEvaluationRequest],
    total_length: int,
) -> Optional[EvaluationSpec]:
    base = cfg.evaluation
    if base is None and request_eval is None:
        return None

    if base is not None:
        spec = EvaluationSpec(
            metric_ids=list(base.metric_ids),
            y_true_key=base.y_true_key,
            compute_train=base.compute_train,
            compute_valid=base.compute_valid,
            train_indices=base.train_indices,
            test_indices=base.test_indices,
        )
    else:
        spec = EvaluationSpec(metric_ids=[], y_true_key="y")

    if request_eval is not None:
        spec.metric_ids = list(request_eval.metric_ids)
        spec.y_true_key = request_eval.y_true_key
        spec.compute_train = request_eval.compute_train
        spec.compute_valid = request_eval.compute_valid
        spec.train_indices = request_eval.train_indices
        spec.test_indices = request_eval.test_indices

    if not spec.metric_ids:
        return None

    if not spec.train_indices or not spec.test_indices:
        split = max(1, min(total_length, int(total_length * 0.8)))
        spec.train_indices = list(range(split))
        spec.test_indices = list(range(split, total_length))

    bindings: Dict[str, ExecutionBinding] = {}
    for mid in spec.metric_ids:
        try:
            bindings[mid] = binding_registry.get_metric_binding(mid)
        except Exception:
            continue
    spec.bindings = bindings
    return spec


@app.get("/health", tags=["system"])
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/registry/configs", response_model=ConfigRegistrationResponse, status_code=status.HTTP_201_CREATED, tags=["configs"])
def register_configuration(payload: ConfigRegistrationRequest, db: RegistryDatabase = Depends(get_database)) -> ConfigRegistrationResponse:
    try:
        cfg = load_model_configuration_from_dict(payload.config, env=payload.env, model_id=payload.model_id)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    spec = _register_model_spec_from_configuration(cfg)
    db.upsert_config(spec.model_id, payload.env, cfg.raw)

    response_spec = {
        "model_id": spec.model_id,
        "name": spec.name,
        "version": spec.version,
        "kind": spec.kind.value,
        "hyperparams_schema": spec.hyperparams.to_json_schema(),
        "tags": spec.tags,
        "metadata": spec.metadata,
    }
    return ConfigRegistrationResponse(model_id=spec.model_id, environment=payload.env, spec=response_spec)


@app.post("/registry/models", response_model=ModelSpecResponse, status_code=status.HTTP_201_CREATED, tags=["models"])
def register_model_spec(payload: ModelSpecRequest) -> ModelSpecResponse:
    spec = _register_simple_model_spec(payload)
    return ModelSpecResponse(
        model_id=spec.model_id,
        name=spec.name,
        version=spec.version,
        kind=spec.kind.value,
        hyperparams=spec.hyperparams.to_json_schema(),
        tags=spec.tags,
        metadata=spec.metadata,
    )


@app.get("/registry/models", tags=["models"])
def list_model_specs(db: RegistryDatabase = Depends(get_database)) -> List[Dict[str, Any]]:
    return db.list_specs()


@app.get("/registry/models/{model_id}", tags=["models"])
def get_model_spec(model_id: str, db: RegistryDatabase = Depends(get_database)) -> Dict[str, Any]:
    spec = db.fetch_spec(model_id)
    if spec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model '{model_id}' not found")
    configs = db.list_configs(model_id)
    spec["configs"] = configs
    return spec


@app.get("/registry/models/{model_id}/configs", tags=["configs"])
def list_model_configs(model_id: str, env: Optional[str] = None, db: RegistryDatabase = Depends(get_database)) -> List[Dict[str, Any]]:
    configs = db.list_configs(model_id, environment=env)
    if not configs:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No configs found for model '{model_id}'")
    return configs


def _execute_async_run(run_id: str, cfg: ModelConfiguration, payload: RunRequest) -> None:
    run_manager.mark_running(run_id)
    try:
        result = _perform_model_run(cfg, payload)
    except Exception as exc:  # pragma: no cover - background execution
        run_manager.mark_failed(run_id, str(exc))
    else:
        run_manager.mark_succeeded(run_id, result)


def _job_to_status(job: AsyncRunJob) -> RunStatusResponse:
    return RunStatusResponse(
        run_id=job.run_id,
        status=job.status.value,
        submitted_at=job.submitted_at.isoformat(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        result=job.result,
        error=job.error,
    )


@app.post("/runs", response_model=RunSubmissionResponse, tags=["runs"], status_code=status.HTTP_202_ACCEPTED)
def submit_run(payload: RunRequest, db: RegistryDatabase = Depends(get_database)) -> RunSubmissionResponse:
    stored_config = db.get_config(payload.model_id, payload.env)
    if stored_config is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Configuration not found for model '{payload.model_id}'")

    try:
        cfg = load_model_configuration_from_dict(
            stored_config,
            env=payload.env,
            model_id=payload.model_id,
            base_path=Path.cwd(),
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid configuration payload: {exc}") from exc

    _register_model_spec_from_configuration(cfg)

    run_id = str(uuid.uuid4())
    run_manager.create_job(run_id)
    payload_copy = RunRequest(**payload.dict())
    run_executor.submit(_execute_async_run, run_id, cfg, payload_copy)
    return RunSubmissionResponse(run_id=run_id, status=AsyncRunStatus.QUEUED.value)


@app.get("/runs/{run_id}", response_model=RunStatusResponse, tags=["runs"])
def get_run_status(run_id: str) -> RunStatusResponse:
    job = run_manager.get_job(run_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Run '{run_id}' not found")
    return _job_to_status(job)


@app.get("/runs", response_model=List[RunStatusResponse], tags=["runs"])
def list_runs() -> List[RunStatusResponse]:
    return [_job_to_status(job) for job in run_manager.list_jobs()]
class AsyncRunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AsyncRunJob:
    run_id: str
    status: AsyncRunStatus
    submitted_at: datetime = field(default_factory=lambda: datetime.utcnow())
    completed_at: Optional[datetime] = None
    result: Optional[RunResponse] = None
    error: Optional[str] = None


class RunManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, AsyncRunJob] = {}
        self._lock = Lock()

    def create_job(self, run_id: str) -> AsyncRunJob:
        job = AsyncRunJob(run_id=run_id, status=AsyncRunStatus.QUEUED)
        with self._lock:
            self._jobs[run_id] = job
        return job

    def mark_running(self, run_id: str) -> None:
        with self._lock:
            job = self._jobs.get(run_id)
            if job:
                job.status = AsyncRunStatus.RUNNING

    def mark_succeeded(self, run_id: str, result: RunResponse) -> None:
        with self._lock:
            job = self._jobs.get(run_id)
            if job:
                job.status = AsyncRunStatus.SUCCEEDED
                job.result = result
                job.completed_at = datetime.utcnow()

    def mark_failed(self, run_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(run_id)
            if job:
                job.status = AsyncRunStatus.FAILED
                job.error = error
                job.completed_at = datetime.utcnow()

    def get_job(self, run_id: str) -> Optional[AsyncRunJob]:
        with self._lock:
            return self._jobs.get(run_id)

    def list_jobs(self) -> List[AsyncRunJob]:
        with self._lock:
            return list(self._jobs.values())
