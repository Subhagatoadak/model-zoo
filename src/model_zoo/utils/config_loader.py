"""Utilities for loading model configuration files with hyperparameters,
variables, transforms, and grouping metadata.

The loader consumes machine-friendly JSON documents so that both humans and
LLMs can edit configurations deterministically. Expected top-level keys:

  {
    "model": {...},
    "hyperparameters": {
      "schema": [...],
      "values": [...]
    },
    "variables": {
      "definitions": [...],
      "groups": [...]
    },
    "evaluation": {...},
    "context": {...}
  }

Fields are optional but recommended. Environment-specific overrides can be
provided by including an "env" key on individual entries. Consumers pass the
target environment when requesting the configuration so the loader filters the
relevant rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json

from .config_tables import hyperparam_space_from_master, params_from_engine
from ..contracts.hyperparams import HyperParamSpace


def _matches_env(entry_env: Optional[str], target_env: Optional[str]) -> bool:
    if entry_env is None or entry_env == "":
        return True
    if target_env is None:
        return False
    if isinstance(entry_env, str):
        parts = [p.strip() for p in entry_env.split(",") if p.strip()]
    else:
        parts = list(entry_env)
    target = str(target_env).strip()
    return target in parts


@dataclass
class TransformSpec:
    """Transformation reference defined in configuration."""

    transform_id: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TransformSpec":
        if not payload:
            raise ValueError("Transform specification cannot be empty")
        transform_id = str(payload.get("id") or payload.get("name") or "").strip()
        if not transform_id:
            raise ValueError("Transform specification missing 'id'")
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("Transform 'params' must be a mapping")
        return cls(transform_id=transform_id, params=params)


@dataclass
class VariableDefinition:
    """Single model variable definition, optionally with transforms."""

    name: str
    source: str
    include: bool = True
    transforms: List[TransformSpec] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VariableDefinition":
        name = str(payload.get("name") or payload.get("id") or "").strip()
        if not name:
            raise ValueError("Variable definition missing 'name'")
        source = str(payload.get("source") or payload.get("from") or "").strip()
        if not source:
            raise ValueError(f"Variable '{name}' missing 'source'")
        include = bool(payload.get("include", True))
        transforms_payload = payload.get("transforms") or []
        transforms: List[TransformSpec] = []
        for item in transforms_payload:
            transforms.append(TransformSpec.from_dict(item))
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"Variable '{name}' metadata must be a mapping")
        return cls(name=name, source=source, include=include, transforms=transforms, metadata=metadata)


@dataclass
class VariableGroup:
    """Optional grouping of variables with its own transforms and output name."""

    group_id: str
    variables: List[str]
    transforms: List[TransformSpec] = field(default_factory=list)
    output_name: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VariableGroup":
        group_id = str(payload.get("id") or payload.get("group_id") or "").strip()
        if not group_id:
            raise ValueError("Variable group missing 'id'")
        variables = payload.get("variables") or payload.get("members") or []
        if not isinstance(variables, Iterable) or isinstance(variables, (str, bytes)):
            raise ValueError(f"Variable group '{group_id}' requires a list of variable names")
        var_list = [str(v).strip() for v in variables if str(v).strip()]
        if not var_list:
            raise ValueError(f"Variable group '{group_id}' must list at least one variable")
        transforms_payload = payload.get("transforms") or []
        transforms: List[TransformSpec] = [TransformSpec.from_dict(item) for item in transforms_payload]
        output_name = payload.get("output_name") or payload.get("as")
        if isinstance(output_name, str):
            output_name = output_name.strip() or None
        else:
            output_name = None
        return cls(group_id=group_id, variables=var_list, transforms=transforms, output_name=output_name)


@dataclass
class VariableConfig:
    definitions: Dict[str, VariableDefinition]
    groups: List[VariableGroup]

    def active_variables(self) -> List[VariableDefinition]:
        return [var for var in self.definitions.values() if var.include]


@dataclass
class EvaluationConfig:
    metric_ids: List[str]
    y_true_key: str
    train_indices: Optional[List[int]] = None
    test_indices: Optional[List[int]] = None
    compute_train: bool = True
    compute_valid: bool = True

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvaluationConfig":
        if not payload:
            raise ValueError("Evaluation configuration cannot be empty")
        metric_ids = payload.get("metric_ids") or []
        if not isinstance(metric_ids, list) or not metric_ids:
            raise ValueError("Evaluation configuration requires 'metric_ids'")
        y_true_key = str(payload.get("y_true_key") or payload.get("target_key") or "").strip()
        if not y_true_key:
            raise ValueError("Evaluation configuration missing 'y_true_key'")
        train_indices = payload.get("train_indices")
        test_indices = payload.get("test_indices")
        compute_train = bool(payload.get("compute_train", True))
        compute_valid = bool(payload.get("compute_valid", True))
        return cls(
            metric_ids=[str(mid).strip() for mid in metric_ids if str(mid).strip()],
            y_true_key=y_true_key,
            train_indices=train_indices,
            test_indices=test_indices,
            compute_train=compute_train,
            compute_valid=compute_valid,
        )


@dataclass
class ModelConfiguration:
    """Loaded configuration ready for downstream consumption."""

    raw: Dict[str, Any]
    model: Dict[str, Any]
    hyperparam_space: HyperParamSpace
    params: Dict[str, Any]
    variables: VariableConfig
    evaluation: Optional[EvaluationConfig]
    context: Dict[str, Any]


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Top-level configuration payload must be a mapping")
    return data


def _extract_variables(payload: Dict[str, Any], env: Optional[str]) -> VariableConfig:
    section = payload.get("variables") or {}
    defs_payload = section.get("definitions") or []
    groups_payload = section.get("groups") or []
    definitions: Dict[str, VariableDefinition] = {}
    for row in defs_payload:
        if not _matches_env(row.get("env"), env):
            continue
        var = VariableDefinition.from_dict(row)
        definitions[var.name] = var
    groups: List[VariableGroup] = []
    for row in groups_payload:
        if not _matches_env(row.get("env"), env):
            continue
        groups.append(VariableGroup.from_dict(row))
    return VariableConfig(definitions=definitions, groups=groups)


def _extract_evaluation(payload: Dict[str, Any], env: Optional[str]) -> Optional[EvaluationConfig]:
    section = payload.get("evaluation")
    if isinstance(section, list):
        for row in section:
            if _matches_env(row.get("env"), env):
                return EvaluationConfig.from_dict(row)
        return None
    if isinstance(section, dict):
        if not section:
            return None
        if not _matches_env(section.get("env"), env):
            return None
        return EvaluationConfig.from_dict(section)
    return None


def _load_model_configuration_from_payload(payload: Dict[str, Any], env: Optional[str], model_id: Optional[str]) -> ModelConfiguration:
    model_section = payload.get("model") or {}
    config_model_id = model_section.get("id") or model_section.get("model_id")
    if model_id is None and config_model_id is None:
        raise ValueError("Model id must be set either in configuration or via argument")
    final_model_id = str(model_id or config_model_id)
    model_section["id"] = final_model_id

    hyper_section = payload.get("hyperparameters") or {}
    master_rows = [row for row in hyper_section.get("schema", []) if _matches_env(row.get("env"), env)]
    if not master_rows:
        raise ValueError("Configuration requires at least one hyperparameter schema row")
    space = hyperparam_space_from_master(master_rows)

    engine_rows = [row for row in hyper_section.get("values", []) if _matches_env(row.get("env"), env)]
    params = params_from_engine(engine_rows, model_id=final_model_id, master=space, env=env)

    variables = _extract_variables(payload, env)
    evaluation = _extract_evaluation(payload, env)

    context = payload.get("context") or {}
    if not isinstance(context, dict):
        raise ValueError("Context section must be a mapping")

    return ModelConfiguration(
        raw=payload,
        model=model_section,
        hyperparam_space=space,
        params=params,
        variables=variables,
        evaluation=evaluation,
        context=context,
    )


def load_model_configuration(path: str | Path, env: Optional[str] = None, model_id: Optional[str] = None) -> ModelConfiguration:
    """Load a configuration document and produce a `ModelConfiguration`.

    Args:
        path: filesystem path to the JSON configuration file.
        env: optional environment name used to filter entries that declare an 'env'.
        model_id: optional explicit model identifier. Defaults to the id from config.
    """

    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved_path}")
    payload = _load_json(resolved_path)
    return _load_model_configuration_from_payload(payload, env=env, model_id=model_id)


def load_model_configuration_from_dict(payload: Dict[str, Any], env: Optional[str] = None, model_id: Optional[str] = None) -> ModelConfiguration:
    """Load a configuration from an in-memory dictionary."""

    if not isinstance(payload, dict):
        raise ValueError("Configuration payload must be a mapping")
    return _load_model_configuration_from_payload(dict(payload), env=env, model_id=model_id)
