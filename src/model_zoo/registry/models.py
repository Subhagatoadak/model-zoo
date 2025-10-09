from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
from enum import Enum
from ..contracts.hyperparams import HyperParamSpace

class ModelKind(Enum):
    """The high‑level mechanism used to execute a model.

    - API: the model is invoked via a remote API/HTTP endpoint.
    - LIBRARY: the model is an in‑process callable (Python module/function).
    - PIPELINE: the model is scheduled/orchestrated by a pipeline system.
    """

    API = "api"
    LIBRARY = "library"
    PIPELINE = "pipeline"

@dataclass
class ModelSpec:
    """A declarative specification for a model entry in the registry.

    This dataclass carries only metadata and a machine‑readable hyperparameter
    schema. It does not contain execution code. How to execute the model is
    described separately via an ExecutionBinding at run time.

    Attributes:
        model_id: Unique identifier for the model (key in the registry).
        name: Human‑friendly, display name of the model.
        version: Semver or revision string for the model spec.
        kind: The mechanism used to execute (see ModelKind).
        hyperparams: Hyperparameter space/schema for validation and tooling.
        tags: Optional list of tags for discovery/organization.
        metadata: Arbitrary, implementation‑specific metadata.
    """
    model_id: str
    name: str
    version: str
    kind: ModelKind
    hyperparams: HyperParamSpace
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this spec into a JSON‑serializable dictionary.

        Returns a minimal, stable representation suitable for audit logs,
        APIs, or storage. Hyperparameters are exported as a JSON Schema‑like
        object via ``HyperParamSpace.to_json_schema()``.
        """
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
    """In‑memory registry for model specifications.

    The registry is a simple map of ``model_id`` to ``ModelSpec``. It offers
    CRUD operations and does not perform execution. Multiple instances can be
    used for isolation in tests or different application domains.
    """

    def __init__(self) -> None:
        """Create an empty registry."""
        self._models: Dict[str, ModelSpec] = {}

    def register(self, spec: ModelSpec) -> None:
        """Register a new model spec.

        Raises:
            KeyError: if a model with the same ``model_id`` already exists.
        """
        if spec.model_id in self._models:
            raise KeyError(f"Model '{spec.model_id}' already exists. Use replace().")
        self._models[spec.model_id] = spec

    def replace(self, model_id: str, spec: ModelSpec) -> None:
        """Replace or upsert a model spec under ``model_id``.

        This is idempotent and will overwrite an existing entry if present.
        """
        self._models[model_id] = spec

    def remove(self, model_id: str) -> None:
        """Remove a model spec by id. Does nothing if absent."""
        self._models.pop(model_id, None)

    def get(self, model_id: str) -> ModelSpec:
        """Fetch a model spec by id.

        Raises:
            KeyError: if ``model_id`` is not found.
        """
        if model_id not in self._models:
            raise KeyError(f"Unknown model_id: {model_id}")
        return self._models[model_id]

    def list(self) -> List[ModelSpec]:
        """Return all registered model specs as a list."""
        return list(self._models.values())
