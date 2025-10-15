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
import csv

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
class CoefficientDefinition:
    name: Optional[str]
    distribution: str
    params: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)

    def as_prior_dict(self, include_name: bool = True) -> Dict[str, Any]:
        data = {"distribution": self.distribution, "params": dict(self.params)}
        if include_name and self.name is not None:
            data["name"] = self.name
        if self.constraints:
            data["constraints"] = dict(self.constraints)
        return data


@dataclass
class CoefficientConfig:
    intercept: Optional[CoefficientDefinition]
    variables: Dict[str, CoefficientDefinition]

    def to_runner_params(self) -> Dict[str, Any]:
        priors: Dict[str, Any] = {}
        constraints: Dict[str, Any] = {}
        if self.intercept:
            priors["intercept"] = self.intercept.as_prior_dict(include_name=False)
            if self.intercept.constraints:
                constraints["intercept"] = dict(self.intercept.constraints)
        if self.variables:
            priors["variables"] = [coeff.as_prior_dict() for coeff in self.variables.values()]
            var_constraints = {
                name: dict(coeff.constraints)
                for name, coeff in self.variables.items()
                if coeff.constraints
            }
            if var_constraints:
                constraints["variables"] = var_constraints
        runner_params: Dict[str, Any] = {}
        if priors:
            runner_params["coefficient_priors"] = priors
        if constraints:
            runner_params["coefficient_constraints"] = constraints
        return runner_params


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
    coefficients: CoefficientConfig
    evaluation: Optional[EvaluationConfig]
    context: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)


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


def _select_env_entry(data: Any, env: Optional[str]) -> Optional[Dict[str, Any]]:
    if isinstance(data, list):
        fallback = None
        for item in data:
            if not isinstance(item, dict):
                continue
            if _matches_env(item.get("env"), env):
                return item
            if fallback is None and _matches_env(item.get("env"), None):
                fallback = item
        return fallback
    if isinstance(data, dict) and _matches_env(data.get("env"), env):
        return data
    return None


def _undot_keys(payload: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in payload.items():
        if "." in key:
            parts = key.split(".")
            cursor = result
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = value
        else:
            result[key] = value
    return result


def _clean_mapping(data: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in data.items():
        if value in ("", None):
            continue
        if isinstance(value, dict):
            nested = _clean_mapping(value)
            if nested:
                cleaned[key] = nested
        elif isinstance(value, list):
            cleaned_list = [item for item in value if item not in ("", None)]
            if cleaned_list:
                cleaned[key] = cleaned_list
        else:
            cleaned[key] = value
    return cleaned


def _parse_coefficient_definition(payload: Dict[str, Any], require_name: bool) -> CoefficientDefinition:
    payload = _undot_keys(payload)
    name = payload.get("name")
    if require_name:
        if not name or not str(name).strip():
            raise ValueError("Coefficient variable definitions require a 'name'")
        name = str(name).strip()
    else:
        name = None
    distribution = str(payload.get("distribution") or "normal").strip().lower()
    params_payload = payload.get("params")
    if isinstance(params_payload, dict):
        params = _clean_mapping(dict(params_payload))
    else:
        params = {
            k: v
            for k, v in payload.items()
            if k not in {"name", "distribution", "env", "constraints"}
            and not isinstance(v, dict)  # skip nested structures like params/constraints
        }
    params = _clean_mapping(params)
    constraints_payload = payload.get("constraints") or {}
    if not isinstance(constraints_payload, dict):
        raise ValueError("Coefficient constraints must be provided as a mapping")
    constraints = _clean_mapping(dict(constraints_payload))
    return CoefficientDefinition(name=name, distribution=distribution, params=params, constraints=constraints)


def _load_table_content(entry: Dict[str, Any], base_path: Optional[Path]) -> List[Dict[str, Any]]:
    if "content" in entry and entry["content"] is not None:
        data = entry["content"]
        if isinstance(data, list):
            return [dict(row) for row in data if isinstance(row, dict)]
        raise ValueError("Table content must be a list of mappings")
    path_value = entry.get("path")
    if not path_value:
        raise ValueError("Coefficient table entry requires either 'content' or 'path'")
    path_obj = Path(path_value)
    if not path_obj.is_absolute():
        if base_path is not None:
            path_obj = (base_path / path_obj).resolve()
        else:
            path_obj = Path.cwd() / path_obj
    if not path_obj.exists():
        raise FileNotFoundError(f"Coefficient table not found: {path_obj}")
    fmt = str(entry.get("format") or path_obj.suffix.lstrip(".") or "csv").lower()
    if fmt == "csv":
        with path_obj.open(newline="", encoding=entry.get("encoding", "utf-8")) as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]
    if fmt in ("json", "jsonl"):
        data = json.loads(path_obj.read_text(encoding=entry.get("encoding", "utf-8")))
        if isinstance(data, list):
            return [dict(row) for row in data if isinstance(row, dict)]
        raise ValueError("JSON coefficient table must decode to a list of mappings")
    raise ValueError(f"Unsupported coefficient table format: {fmt}")


def _extract_coefficients(payload: Dict[str, Any], env: Optional[str], base_path: Optional[Path]) -> CoefficientConfig:
    section = payload.get("coefficient_priors") or payload.get("coefficients") or {}
    intercept_entry = section.get("intercept")
    intercept = None
    if intercept_entry is not None:
        chosen = _select_env_entry(intercept_entry, env)
        if chosen:
            intercept = _parse_coefficient_definition(chosen, require_name=False)

    variables_entries = section.get("variables") or []
    variables: Dict[str, CoefficientDefinition] = {}
    if isinstance(variables_entries, dict):
        variables_entries = [variables_entries]
    if isinstance(variables_entries, list):
        for raw in variables_entries:
            if not isinstance(raw, dict):
                continue
            if not _matches_env(raw.get("env"), env):
                continue
            coeff = _parse_coefficient_definition(raw, require_name=True)
            variables[coeff.name] = coeff
    tables = section.get("tables") or []
    for table_entry in tables:
        if not isinstance(table_entry, dict):
            continue
        if not _matches_env(table_entry.get("env"), env):
            continue
        rows = _load_table_content(table_entry, base_path)
        for row in rows:
            coeff = _parse_coefficient_definition(row, require_name=True)
            variables[coeff.name] = coeff
    return CoefficientConfig(intercept=intercept, variables=variables)


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


def _as_dependency_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        result: List[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("Dependency entries must be strings")
            item_clean = item.strip()
            if item_clean:
                result.append(item_clean)
        return result
    raise ValueError("Dependencies must be strings or lists of strings")


def _extract_dependencies(payload: Dict[str, Any], env: Optional[str]) -> List[str]:
    raw = payload.get("dependencies")
    if raw in (None, ""):
        return []

    collected: List[str] = []

    def extend(values: List[str]) -> None:
        for val in values:
            if val not in collected:
                collected.append(val)

    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, str):
                extend(_as_dependency_list(entry))
                continue
            if isinstance(entry, dict):
                if not _matches_env(entry.get("env"), env):
                    continue
                if "packages" in entry:
                    extend(_as_dependency_list(entry.get("packages")))
                elif "dependencies" in entry:
                    extend(_as_dependency_list(entry.get("dependencies")))
                elif "package" in entry:
                    extend(_as_dependency_list(entry.get("package")))
                else:
                    raise ValueError("Dependency dict entries require 'packages' or 'package'")
                continue
            raise ValueError("Dependencies list may contain strings or mappings with env/packages")
    elif isinstance(raw, dict):
        for key, value in raw.items():
            key_lower = str(key).strip().lower()
            if key_lower in {"all", "base", "default", "*"}:
                extend(_as_dependency_list(value))
                continue
            if key_lower == "envs":
                env_entries = value
                if not isinstance(env_entries, list):
                    raise ValueError("dependencies.envs must be a list of mappings")
                for entry in env_entries:
                    if not isinstance(entry, dict):
                        raise ValueError("dependencies.envs entries must be mappings")
                    if not _matches_env(entry.get("env"), env):
                        continue
                    extend(_as_dependency_list(entry.get("packages") or entry.get("dependencies") or entry.get("package")))
                continue
            if not _matches_env(key, env):
                continue
            extend(_as_dependency_list(value))
    else:
        extend(_as_dependency_list(raw))

    return collected


def _load_model_configuration_from_payload(payload: Dict[str, Any], env: Optional[str], model_id: Optional[str], base_path: Optional[Path] = None) -> ModelConfiguration:
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
    for table_entry in hyper_section.get("tables", []) or []:
        if not isinstance(table_entry, dict):
            continue
        if not _matches_env(table_entry.get("env"), env):
            continue
        table_rows = _load_table_content(table_entry, base_path)
        engine_rows.extend(table_rows)
    params = params_from_engine(engine_rows, model_id=final_model_id, master=space, env=env)

    variables = _extract_variables(payload, env)
    coefficients = _extract_coefficients(payload, env, base_path)
    evaluation = _extract_evaluation(payload, env)

    context = payload.get("context") or {}
    if not isinstance(context, dict):
        raise ValueError("Context section must be a mapping")

    dependencies = _extract_dependencies(payload, env)

    return ModelConfiguration(
        raw=payload,
        model=model_section,
        hyperparam_space=space,
        params=params,
        variables=variables,
        coefficients=coefficients,
        evaluation=evaluation,
        context=context,
        dependencies=dependencies,
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
    return _load_model_configuration_from_payload(payload, env=env, model_id=model_id, base_path=resolved_path.parent)


def load_model_configuration_from_dict(payload: Dict[str, Any], env: Optional[str] = None, model_id: Optional[str] = None, base_path: Optional[str | Path] = None) -> ModelConfiguration:
    """Load a configuration from an in-memory dictionary."""

    if not isinstance(payload, dict):
        raise ValueError("Configuration payload must be a mapping")
    path_obj = Path(base_path) if base_path is not None else None
    return _load_model_configuration_from_payload(dict(payload), env=env, model_id=model_id, base_path=path_obj)
