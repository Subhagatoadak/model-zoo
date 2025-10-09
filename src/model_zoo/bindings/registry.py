from __future__ import annotations
"""Binding registry for resolving execution bindings by id.

This keeps bindings outside registries. Applications can implement their own
source (DB/config) by subclassing `BindingRegistry`.
"""
from typing import Optional, Dict, Any


class BindingRegistry:
    """Abstract binding registry.

    Implementations should resolve model and metric bindings by their ids.
    """

    def get_model_binding(self, model_id: str) -> Optional[Any]:
        raise NotImplementedError

    def get_metric_binding(self, metric_id: str) -> Optional[Any]:
        raise NotImplementedError


class InMemoryBindingRegistry(BindingRegistry):
    """Simple in-memory binding registry suitable for demos/tests."""

    def __init__(self) -> None:
        self._model_bindings: Dict[str, Any] = {}
        self._metric_bindings: Dict[str, Any] = {}

    def register_model_binding(self, model_id: str, binding: Any) -> None:
        self._model_bindings[model_id] = binding

    def register_metric_binding(self, metric_id: str, binding: Any) -> None:
        self._metric_bindings[metric_id] = binding

    def get_model_binding(self, model_id: str) -> Optional[Any]:
        return self._model_bindings.get(model_id)

    def get_metric_binding(self, metric_id: str) -> Optional[Any]:
        return self._metric_bindings.get(metric_id)

