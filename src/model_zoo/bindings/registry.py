from __future__ import annotations
"""Binding registry for resolving execution bindings by id.

This keeps bindings outside registries. Applications can implement their own
source (DB/config) by subclassing `BindingRegistry`.
"""
from typing import Optional, Dict, Any
from threading import Lock


class BindingRegistry:
    """Abstract binding registry.

    Implementations should resolve model and metric bindings by their ids.
    """

    def get_model_binding(self, model_id: str) -> Optional[Any]:
        """Return the execution binding for a model_id, or None if missing."""
        raise NotImplementedError

    def get_metric_binding(self, metric_id: str) -> Optional[Any]:
        """Return the execution binding for a metric_id, or None if missing."""
        raise NotImplementedError


class InMemoryBindingRegistry(BindingRegistry):
    """Simple in-memory binding registry suitable for demos/tests."""

    def __init__(self) -> None:
        self._model_bindings: Dict[str, Any] = {}
        self._metric_bindings: Dict[str, Any] = {}
        self._lock = Lock()

    def register_model_binding(self, model_id: str, binding: Any) -> None:
        """Register or replace the binding for a given model_id."""
        with self._lock:
            self._model_bindings[model_id] = binding

    def register_metric_binding(self, metric_id: str, binding: Any) -> None:
        """Register or replace the binding for a given metric_id."""
        with self._lock:
            self._metric_bindings[metric_id] = binding

    def get_model_binding(self, model_id: str) -> Optional[Any]:
        """Return the binding registered for model_id, if any."""
        with self._lock:
            return self._model_bindings.get(model_id)

    def get_metric_binding(self, metric_id: str) -> Optional[Any]:
        """Return the binding registered for metric_id, if any."""
        with self._lock:
            return self._metric_bindings.get(metric_id)
