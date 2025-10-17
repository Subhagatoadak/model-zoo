"""Utilities to upgrade legacy configuration payloads into the canonical schema.

The Model Zoo registry expects configuration documents to conform to the
``config_template.json`` structure. Some external tools emit simpler payloads
that only describe algorithm knobs and variable metadata. This module provides
helpers to coerce those documents into a shape that the existing loader can
understand so they can still be registered and reasoned about.
"""
from __future__ import annotations

import copy
import csv
import io
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class LegacyUpgradeResult:
    payload: Dict[str, Any]
    converted: bool


def upgrade_legacy_payload(
    payload: Dict[str, Any],
    model_id_hint: Optional[str] = None,
    base_path: Optional[Path] = None,
) -> LegacyUpgradeResult:
    """Return a configuration payload compatible with the canonical loader.

    Args:
        payload: Raw configuration dictionary provided by a client.
        model_id_hint: Optional identifier supplied alongside the payload.
        base_path: Optional directory used to resolve relative file references.
    """

    if _looks_canonical(payload):
        return LegacyUpgradeResult(payload=dict(payload), converted=False)

    original = copy.deepcopy(payload)

    algorithm = payload.get("algorithm")
    error_allocation = payload.get("errorallocation") or payload.get("error_allocation")
    spec = payload.get("spec") or {}
    params = payload.get("params") or {}

    flattened_params = _flatten_params(params)

    model_id = _select_model_id(model_id_hint, spec)
    model_section = _build_model_section(model_id, spec, algorithm, error_allocation)

    hyperparameters = _build_hyperparameters(flattened_params, model_id)

    variable_rows = _load_variable_rows(payload, base_path)
    variable_definitions = _build_variable_definitions(variable_rows)
    coefficient_priors = _build_coefficient_priors(variable_rows, spec)

    context_payload: Dict[str, Any] = {
        "legacy": {
            "source": "unstructured",
            "algorithm": algorithm,
            "error_allocation": error_allocation,
            "spec": spec,
        }
    }
    if variable_rows:
        context_payload["legacy"]["variable_config_rows"] = variable_rows
    if flattened_params:
        context_payload["legacy"]["parameters"] = flattened_params
    context_payload["legacy"]["original_payload"] = original

    converted_payload: Dict[str, Any] = {
        "model": model_section,
        "hyperparameters": hyperparameters,
        "variables": {
            "definitions": variable_definitions,
            "groups": [],
        },
        "coefficient_priors": coefficient_priors,
        "metrics": [],
        "context": context_payload,
    }

    return LegacyUpgradeResult(payload=converted_payload, converted=True)


def _looks_canonical(payload: Dict[str, Any]) -> bool:
    if "model" in payload or "hyperparameters" in payload:
        return True
    variables = payload.get("variables")
    if isinstance(variables, dict) and ("definitions" in variables or "groups" in variables):
        return True
    return False


def _select_model_id(model_id_hint: Optional[str], spec: Dict[str, Any]) -> str:
    candidate = model_id_hint or spec.get("model_id") or spec.get("id") or spec.get("dep_var")
    if not candidate:
        return "legacy_model"
    return str(candidate)


def _build_model_section(model_id: str, spec: Dict[str, Any], algorithm: Any, error_allocation: Any) -> Dict[str, Any]:
    algorithm_type = None
    if isinstance(algorithm, dict):
        raw_type = algorithm.get("type") or algorithm.get("name")
        if raw_type:
            algorithm_type = str(raw_type)
    name = spec.get("model_name") or spec.get("name")
    if not name:
        if algorithm_type:
            cleaned = " ".join(str(algorithm_type).replace("_", " ").split())
            name = f"{cleaned.title()} Model"
        else:
            name = str(model_id)
    metadata = {
        "legacy_spec": spec,
        "legacy_algorithm": algorithm,
        "legacy_error_allocation": error_allocation,
    }
    tags: List[str] = ["legacy-import"]
    if spec.get("prior"):
        tags.append("uses-priors")
    return {
        "id": model_id,
        "name": str(name),
        "version": str(spec.get("version") or "0.0.1"),
        "kind": str(spec.get("kind") or "pipeline"),
        "metadata": metadata,
        "tags": tags,
    }


def _flatten_params(data: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(data, dict):
        result: Dict[str, Any] = {}
        for key, value in data.items():
            key_str = str(key)
            next_prefix = f"{prefix}.{key_str}" if prefix else key_str
            result.update(_flatten_params(value, next_prefix))
        return result
    if isinstance(data, list):
        result: Dict[str, Any] = {}
        for idx, value in enumerate(data):
            result.update(_flatten_params(value, f"{prefix}[{idx}]"))
        return result
    return {prefix: data}


def _build_hyperparameters(flattened: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    schema_rows: List[Dict[str, Any]] = []
    value_rows: List[Dict[str, Any]] = []
    for name, value in flattened.items():
        if not name:
            continue
        dtype = _infer_dtype(value)
        schema_rows.append(
            {
                "name": name,
                "dtype": dtype,
                "default": value,
                "description": "Imported from legacy params",
            }
        )
        value_rows.append(
            {
                "model_id": model_id,
                "param": name,
                "value": value,
            }
        )
    return {
        "schema": schema_rows,
        "values": value_rows,
    }


def _infer_dtype(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    return "string"


def _load_variable_rows(payload: Dict[str, Any], base_path: Optional[Path]) -> List[Dict[str, Any]]:
    source = (
        payload.get("variable_config")
        or payload.get("variable_config_csv")
        or payload.get("variableConfig")
        or payload.get("variables_csv")
        or payload.get("variables_table")
    )
    if source is None:
        return []
    if isinstance(source, str):
        return _read_csv_rows(source)
    if isinstance(source, list):
        return [_clean_row(dict(row)) for row in source if isinstance(row, dict)]
    if isinstance(source, dict):
        if "content" in source and source["content"] is not None:
            return _read_csv_rows(str(source["content"]))
        if "path" in source:
            path_obj = Path(str(source["path"]))
            if not path_obj.is_absolute() and base_path is not None:
                path_obj = (base_path / path_obj).resolve()
            rows = _read_csv_rows(path_obj.read_text())
            return rows
        if "rows" in source and isinstance(source["rows"], list):
            return [_clean_row(dict(row)) for row in source["rows"] if isinstance(row, dict)]
    return []


def _read_csv_rows(content: str) -> List[Dict[str, Any]]:
    text = content.strip()
    if not text:
        return []
    reader = csv.DictReader(io.StringIO(text))
    rows: List[Dict[str, Any]] = []
    for row in reader:
        rows.append(_clean_row(row))
    return rows


def _clean_row(row: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in row.items():
        key_str = str(key).strip()
        if not key_str:
            continue
        cleaned[key_str.upper()] = _cleanup_value(value)
    return cleaned


def _cleanup_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        lower = stripped.lower()
        if lower in {"true", "yes", "1"}:
            return True
        if lower in {"false", "no", "0"}:
            return False
        try:
            if "." in stripped:
                return float(stripped)
            return int(stripped)
        except ValueError:
            return stripped
    return value


def _build_variable_definitions(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    definitions: List[Dict[str, Any]] = []
    for row in rows:
        name_raw = row.get("COLUMN")
        if not name_raw:
            continue
        name = str(name_raw)
        include_value = row.get("INCLUDE")
        include = True if include_value is None else bool(include_value)
        metadata = _row_to_metadata(row)
        definitions.append(
            {
                "name": name,
                "source": name,
                "include": include,
                "metadata": metadata,
            }
        )
    return definitions


def _row_to_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for key, value in row.items():
        if key == "COLUMN":
            continue
        if value in (None, ""):
            continue
        key_norm = _normalize_key(key)
        metadata[key_norm] = value
    return metadata


def _normalize_key(key: str) -> str:
    key_clean = key.strip().lower().replace(" ", "_")
    key_clean = key_clean.replace("/", "_")
    key_clean = key_clean.replace("__", "_")
    return key_clean


def _build_coefficient_priors(rows: Iterable[Dict[str, Any]], spec: Dict[str, Any]) -> Dict[str, Any]:
    intercept_entry: Optional[Dict[str, Any]] = None
    prior_flag = spec.get("prior")
    prior_weight = spec.get("prior_weight")
    if isinstance(prior_flag, dict):
        intercept_entry = _coerce_prior_entry("intercept", prior_flag)
    else:
        mu_value = None if isinstance(prior_flag, bool) else _safe_float(prior_flag)
        if mu_value is not None:
            intercept_entry = {
                "distribution": "normal",
                "params": {"mu": mu_value},
            }
            sigma_value = _safe_float(prior_weight)
            if sigma_value is not None:
                intercept_entry["params"]["sigma"] = sigma_value

    variable_priors: List[Dict[str, Any]] = []
    for row in rows:
        name_raw = row.get("COLUMN")
        if not name_raw:
            continue
        name = str(name_raw)
        prior_type = _as_optional_str(row.get("PRIOR TYPE"))
        prior_value = _safe_float(row.get("PRIOR"))
        prior_weight_val = _safe_float(row.get("PRIOR WEIGHT"))
        lower = _safe_float(row.get("LOWER BOUND"))
        upper = _safe_float(row.get("UPPER BOUND"))
        sign_expectation = _as_optional_str(row.get("SIGN EXPECTATION"))
        if not any([prior_type, prior_value is not None, prior_weight_val is not None, lower is not None, upper is not None, sign_expectation]):
            continue
        distribution = (prior_type or "normal").strip().lower()
        entry: Dict[str, Any] = {"name": name, "distribution": distribution, "params": {}, "constraints": {}}
        if prior_value is not None and not math.isnan(prior_value):
            entry["params"]["mu"] = prior_value
        if prior_weight_val is not None and not math.isnan(prior_weight_val):
            key = "sigma" if distribution in {"normal", "laplace", "lognormal"} else "scale"
            entry["params"][key] = prior_weight_val
        if sign_expectation:
            entry["constraints"]["sign"] = sign_expectation.lower()
        if lower is not None and not math.isnan(lower):
            entry["constraints"]["lower"] = lower
        if upper is not None and not math.isnan(upper):
            entry["constraints"]["upper"] = upper
        if not entry["constraints"]:
            entry.pop("constraints")
        if not entry["params"]:
            entry.pop("params")
        variable_priors.append(entry)

    result: Dict[str, Any] = {}
    if intercept_entry:
        result["intercept"] = intercept_entry
    if variable_priors:
        result["variables"] = variable_priors
    return result


def _as_optional_str(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return str(value)


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        if math.isinf(result):
            return None
        return result
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        return None
    if math.isinf(result):
        return None
    return result


def _coerce_prior_entry(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    entry = {"distribution": str(payload.get("distribution") or "normal").lower(), "params": {}, "constraints": {}}
    for key in ("mu", "sigma", "scale", "tau"):
        val = _safe_float(payload.get(key))
        if val is not None:
            entry["params"][key] = val
    for key in ("lower", "upper"):
        val = _safe_float(payload.get(key))
        if val is not None:
            entry["constraints"][key] = val
    sign = _as_optional_str(payload.get("sign"))
    if sign:
        entry["constraints"]["sign"] = sign.lower()
    if not entry["params"]:
        entry.pop("params")
    if not entry["constraints"]:
        entry.pop("constraints")
    return entry
