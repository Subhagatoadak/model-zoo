"""Parameter validation service interface.

Provide concrete implementations to enforce HyperParamSpace constraints for
models and metrics before execution.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict
from ..registry.models import ModelSpec
from ..registry.metrics import EvalMetricSpec

class ParamValidationService(ABC):
    """Abstract validator for model and metric parameters."""
    @abstractmethod
    def validate_model_params(self, model_spec: ModelSpec, params: Dict) -> None:
        """Validate parameters for a model against `model_spec.hyperparams`."""
        ...
    @abstractmethod
    def validate_metric_params(self, metric_spec: EvalMetricSpec, params: Dict) -> None:
        """Validate parameters for a metric against `metric_spec.params`."""
        ...
