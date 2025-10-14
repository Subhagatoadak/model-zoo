"""Built-in transformation registry for variable and group processing."""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List
import math
from statistics import mean, pstdev


TransformFn = Callable[[Any, Dict[str, Any]], Any]


def _identity(value: Any, params: Dict[str, Any]) -> Any:
    return value


def _as_float_list(value: Any) -> List[float]:
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, tuple):
        return [float(v) for v in value]
    raise TypeError("Transform requires a list-like value")


def _standardize(value: Any, params: Dict[str, Any]) -> List[float]:
    items = _as_float_list(value)
    if not items:
        return items
    mu = params.get("mean")
    sigma = params.get("std")
    if mu is None:
        mu = mean(items)
    if sigma is None:
        sigma = pstdev(items) or 1.0
    return [(x - float(mu)) / float(sigma) for x in items]


def _min_max_scale(value: Any, params: Dict[str, Any]) -> List[float]:
    items = _as_float_list(value)
    if not items:
        return items
    min_val = params.get("min")
    max_val = params.get("max")
    if min_val is None:
        min_val = min(items)
    if max_val is None:
        max_val = max(items)
    span = float(max_val) - float(min_val)
    if span == 0:
        return [0.0 for _ in items]
    return [(x - float(min_val)) / span for x in items]


def _log1p(value: Any, params: Dict[str, Any]) -> List[float]:
    items = _as_float_list(value)
    return [math.log1p(float(x)) for x in items]


def _clip(value: Any, params: Dict[str, Any]) -> List[float]:
    items = _as_float_list(value)
    lo = params.get("min", float("-inf"))
    hi = params.get("max", float("inf"))
    lo = float(lo)
    hi = float(hi)
    return [min(max(x, lo), hi) for x in items]


def _poly_features(value: Any, params: Dict[str, Any]) -> Dict[str, List[float]]:
    items = _as_float_list(value)
    degree = int(params.get("degree", 2))
    prefix = params.get("prefix", "poly")
    outputs: Dict[str, List[float]] = {}
    for d in range(1, degree + 1):
        key = f"{prefix}_{d}"
        outputs[key] = [float(x) ** d for x in items]
    return outputs


def _stack(values: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(values, dict):
        return values
    if isinstance(values, (list, tuple)):
        keys = params.get("keys")
        if keys:
            if len(keys) != len(values):
                raise ValueError("stack transform requires keys length to match values")
            return {k: v for k, v in zip(keys, values)}
        return {str(i): v for i, v in enumerate(values)}
    raise TypeError("stack transform expects list/tuple or dict input")


DEFAULT_TRANSFORMS: Dict[str, TransformFn] = {
    "identity": _identity,
    "standardize": _standardize,
    "min_max": _min_max_scale,
    "log1p": _log1p,
    "clip": _clip,
    "poly_features": _poly_features,
    "stack": _stack,
}


def apply_transforms(value: Any, specs: Iterable[Dict[str, Any] | tuple[str, Dict[str, Any]] | Any], registry: Dict[str, TransformFn] | None = None) -> Any:
    """Apply a sequence of transform specs to a value.

    Each spec can be either a mapping with keys `id` and optional `params`, or a
    tuple of `(transform_id, params)`.
    """

    if registry is None:
        registry = DEFAULT_TRANSFORMS
    result = value
    for spec in specs:
        if isinstance(spec, tuple):
            transform_id, params = spec
        elif isinstance(spec, dict):
            transform_id = spec.get("id")
            params = spec.get("params", {})
        else:
            transform_id = str(spec)
            params = {}
        if transform_id not in registry:
            raise KeyError(f"Unknown transform id: {transform_id}")
        fn = registry[transform_id]
        result = fn(result, params)
    return result


def merge_transform_registries(custom: Dict[str, TransformFn] | None = None) -> Dict[str, TransformFn]:
    registry = dict(DEFAULT_TRANSFORMS)
    if custom:
        registry.update(custom)
    return registry

