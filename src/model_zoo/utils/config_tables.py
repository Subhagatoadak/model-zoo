from __future__ import annotations
"""Helpers to configure HyperParamSpace and params from tabular configs.

Two-table pattern:
- Master config table (schema): defines hyperparameters and validation hints.
- Engine config table (values): per-model (and optionally per-env) parameter values.

This module avoids heavy dependencies and accepts tables as lists of dicts. You
can add small CSV readers on top of these helpers if needed.
"""
from typing import Any, Dict, List, Optional
import json
from ..contracts.hyperparams import HyperParamField, HyperParamSpace, DType


_DTYPE_MAP = {
    "int": DType.INT, "integer": DType.INT,
    "float": DType.FLOAT, "number": DType.FLOAT,
    "bool": DType.BOOL, "boolean": DType.BOOL,
    "str": DType.STR, "string": DType.STR,
    "dict": DType.DICT, "object": DType.DICT,
    "list": DType.LIST, "array": DType.LIST,
}


def _to_dtype(s: Any) -> DType:
    if isinstance(s, DType):
        return s
    key = str(s).strip().lower()
    if key not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype: {s}")
    return _DTYPE_MAP[key]


def _parse_bool(x: Any) -> Optional[bool]:
    if x is None or x == "":
        return None
    if isinstance(x, bool):
        return x
    v = str(x).strip().lower()
    if v in ("1", "true", "yes", "y"): return True
    if v in ("0", "false", "no", "n"): return False
    return None


def hyperparam_space_from_master(master_rows: List[Dict[str, Any]]) -> HyperParamSpace:
    """Build a HyperParamSpace from a master config table.

    Expected columns per row (case-insensitive, extra keys ignored):
      - name (str) — parameter name (required)
      - dtype (str) — int|float|bool|string|object|array (required)
      - required (bool/str) — if the parameter is required
      - default (any) — default value
      - description (str)
      - min (float) / max (float) — numeric bounds
      - enum (comma-separated) — allowed string/number values
      - pattern (str) — regex for strings
      - example (any)
    """
    fields: Dict[str, HyperParamField] = {}
    for row in master_rows:
        name = row.get("name") or row.get("param")
        if not name:
            raise ValueError("Master row missing 'name' field")
        dtype = _to_dtype(row.get("dtype"))
        required = bool(_parse_bool(row.get("required")) or False)
        default = row.get("default")
        description = row.get("description") or ""
        min_value = row.get("min") or row.get("minimum")
        max_value = row.get("max") or row.get("maximum")
        pattern = row.get("pattern")
        example = row.get("example")
        enum_raw = row.get("enum")
        choices = None
        if enum_raw:
            if isinstance(enum_raw, list):
                choices = enum_raw
            else:
                choices = [v.strip() for v in str(enum_raw).split(",") if v.strip()]
        fields[name] = HyperParamField(
            name=name, dtype=dtype, required=required, default=default,
            description=description or "", choices=choices,
            min_value=float(min_value) if min_value not in (None, "") else None,
            max_value=float(max_value) if max_value not in (None, "") else None,
            pattern=pattern, example=example,
        )
    return HyperParamSpace(fields=fields)


def params_from_engine(
    engine_rows: List[Dict[str, Any]],
    model_id: str,
    master: HyperParamSpace,
    env: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract a params dict for `model_id` (and optionally `env`) from an engine table.

    Expected columns per row (case-insensitive):
      - model_id (str) — target model id
      - param (str) — parameter name
      - value (any) — parameter value as string/number/bool
      - env (optional, str) — environment filter (e.g., dev/stage/prod)

    The function coerces values using the dtype from `master` when present.
    Unknown params are included as-is.
    """
    params: Dict[str, Any] = {}
    # quick name->dtype map from master
    dtype_map = {n: f.dtype for n, f in (master.fields or {}).items()}

    def coerce(name: str, val: Any) -> Any:
        dt = dtype_map.get(name)
        if dt is None:
            return val
        # Handle booleans
        if dt == DType.BOOL:
            b = _parse_bool(val)
            if b is None:
                raise ValueError(f"Cannot parse boolean for {name}: {val}")
            return b
        # Int
        if dt == DType.INT:
            try:
                return int(val)
            except Exception:
                raise ValueError(f"Cannot parse int for {name}: {val}")
        # Float/number
        if dt == DType.FLOAT:
            try:
                return float(val)
            except Exception:
                raise ValueError(f"Cannot parse float for {name}: {val}")
        # Lists via comma-separated
        if dt == DType.LIST and isinstance(val, str):
            s = val.strip()
            # Try JSON array first
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    pass
            # Fallback: comma-separated list
            return [v.strip() for v in s.split(",") if v.strip()]
        # Dicts: parse JSON object if string
        if dt == DType.DICT and isinstance(val, str):
            s = val.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
        return val

    for row in engine_rows:
        rid = row.get("model_id") or row.get("model")
        if str(rid) != str(model_id):
            continue
        if env is not None and str(row.get("env") or "") != str(env):
            continue
        pname = row.get("param") or row.get("name")
        if not pname:
            continue
        params[pname] = coerce(pname, row.get("value"))
    return params
