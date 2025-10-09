from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict
from ..registry.models import ModelSpec
from ..registry.metrics import EvalMetricSpec

class ParamValidationService(ABC):
    @abstractmethod
    def validate_model_params(self, model_spec: ModelSpec, params: Dict) -> None: ...
    @abstractmethod
    def validate_metric_params(self, metric_spec: EvalMetricSpec, params: Dict) -> None: ...
