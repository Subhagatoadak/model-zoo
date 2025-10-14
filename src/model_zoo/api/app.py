"""FastAPI application exposing endpoints to register models and configurations."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

import psycopg
from psycopg.types.json import Json

from ..registry.models import ModelRegistry, ModelSpec, ModelKind
from ..utils.config_loader import (
    load_model_configuration_from_dict,
    ModelConfiguration,
)
from ..utils.env import read_env_file, build_postgres_dsn
from ..bindings.registry import InMemoryBindingRegistry


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
binding_registry = InMemoryBindingRegistry()

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
