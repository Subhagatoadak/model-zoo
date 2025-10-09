from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
from enum import Enum
from ..contracts.hyperparams import HyperParamSpace

class ModelKind(Enum):
    API = "api"
    LIBRARY = "library"
    PIPELINE = "pipeline"

@dataclass
class ModelSpec:
    model_id: str
    name: str
    version: str
    kind: ModelKind
    hyperparams: HyperParamSpace
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "kind": self.kind.value,
            "hyperparams_schema": self.hyperparams.to_json_schema(),
            "tags": self.tags,
            "metadata": self.metadata,
        }

class ModelRegistry:
    def __init__(self) -> None:
        self._models: Dict[str, ModelSpec] = {}

    def register(self, spec: ModelSpec) -> None:
        if spec.model_id in self._models:
            raise KeyError(f"Model '{spec.model_id}' already exists. Use replace().")
        self._models[spec.model_id] = spec

    def replace(self, model_id: str, spec: ModelSpec) -> None:
        self._models[model_id] = spec

    def remove(self, model_id: str) -> None:
        self._models.pop(model_id, None)

    def get(self, model_id: str) -> ModelSpec:
        if model_id not in self._models:
            raise KeyError(f"Unknown model_id: {model_id}")
        return self._models[model_id]

    def list(self) -> List[ModelSpec]:
        return list(self._models.values())
