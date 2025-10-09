from __future__ import annotations
from typing import Any, Dict, List, Optional
import re

from .validation import ParamValidationService
from ..registry.models import ModelSpec
from ..registry.metrics import EvalMetricSpec
from .hyperparams import HyperParamSpace, HyperParamField, DType


class DefaultParamValidator(ParamValidationService):
    """Lightweight, dependency-free validator against HyperParamSpace.

    - Enforces required fields.
    - Validates basic types, enum choices, numeric ranges, and string patterns.
    - Allows additional (unknown) parameters to pass through.
    """

    def validate_model_params(self, model_spec: ModelSpec, params: Dict) -> None:
        self._validate_space(model_spec.hyperparams, params, ref=f"model:{model_spec.model_id}")

    def validate_metric_params(self, metric_spec: EvalMetricSpec, params: Dict) -> None:
        self._validate_space(metric_spec.params, params, ref=f"metric:{metric_spec.metric_id}")

    # Implementation
    def _validate_space(self, space: HyperParamSpace, params: Dict[str, Any], ref: str) -> None:
        errors: List[str] = []
        fields: Dict[str, HyperParamField] = space.fields or {}
        params = params or {}

        # Required
        for name, field in fields.items():
            if field.required and name not in params:
                errors.append(f"Missing required param '{name}'")

        # Known fields validation
        for name, value in params.items():
            if name not in fields:
                # allowed (additionalProperties=true)
                continue
            field = fields[name]
            self._validate_field(field, value, errors)

        if errors:
            raise ValueError(f"Parameter validation failed for {ref}: " + "; ".join(errors))

    def _validate_field(self, field: HyperParamField, value: Any, errors: List[str]) -> None:
        # Type check
        try:
            if field.dtype == DType.INT:
                if not isinstance(value, int):
                    errors.append(f"'{field.name}' expected int, got {type(value).__name__}")
            elif field.dtype == DType.FLOAT:
                if not isinstance(value, (int, float)):
                    errors.append(f"'{field.name}' expected number, got {type(value).__name__}")
            elif field.dtype == DType.BOOL:
                if not isinstance(value, bool):
                    errors.append(f"'{field.name}' expected boolean, got {type(value).__name__}")
            elif field.dtype == DType.STR:
                if not isinstance(value, str):
                    errors.append(f"'{field.name}' expected string, got {type(value).__name__}")
            elif field.dtype == DType.LIST:
                if not isinstance(value, list):
                    errors.append(f"'{field.name}' expected array, got {type(value).__name__}")
            elif field.dtype == DType.DICT:
                if not isinstance(value, dict):
                    errors.append(f"'{field.name}' expected object, got {type(value).__name__}")
        except Exception:
            errors.append(f"'{field.name}' type validation error")

        # Choices
        if field.choices is not None and value is not None:
            try:
                if value not in field.choices:
                    errors.append(f"'{field.name}' must be one of {field.choices}, got {value}")
            except Exception:
                errors.append(f"'{field.name}' enum validation error")

        # Numeric bounds
        if isinstance(value, (int, float)):
            if field.min_value is not None and value < field.min_value:
                errors.append(f"'{field.name}' must be >= {field.min_value}, got {value}")
            if field.max_value is not None and value > field.max_value:
                errors.append(f"'{field.name}' must be <= {field.max_value}, got {value}")

        # String pattern
        if isinstance(value, str) and field.pattern:
            try:
                if not re.fullmatch(field.pattern, value):
                    errors.append(f"'{field.name}' does not match pattern {field.pattern}")
            except re.error:
                # Invalid pattern in schema; skip enforcement
                pass

