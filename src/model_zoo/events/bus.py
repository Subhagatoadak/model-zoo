from __future__ import annotations
"""Event bus interfaces for run lifecycle notifications.

Used to notify external agents or listeners of run submissions, status changes,
and evaluation completions. Default is in-memory (no-op listener storage).
"""
from typing import Any, Dict, List, Optional, Callable


class EventBus:
    """Abstract event bus for publishing lifecycle notifications."""

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        """Publish a topic/payload event. Implementations decide delivery semantics."""
        raise NotImplementedError


class InMemoryEventBus(EventBus):
    """Simple in-memory pub implementation that calls registered listeners."""

    def __init__(self) -> None:
        self._listeners: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}

    def subscribe(self, topic: str, fn: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback to be invoked when `topic` is published."""
        self._listeners.setdefault(topic, []).append(fn)

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        """Invoke all listeners for `topic` with `payload` (best‑effort)."""
        for fn in self._listeners.get(topic, []):
            try:
                fn(payload)
            except Exception:
                # best-effort; listeners shouldn't break publisher
                pass


class WebhookEventBus(EventBus):
    """Webhook publisher (best-effort). Requires `requests` if used.

    In environments without network or requests, this can be avoided or wrapped
    by a try/except to degrade gracefully.
    """

    def __init__(self, endpoint: str, headers: Optional[Dict[str, str]] = None, timeout_s: int = 5) -> None:
        self._endpoint = endpoint
        self._headers = headers or {}
        self._timeout = timeout_s

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        """POST an event to the configured webhook endpoint (best‑effort)."""
        try:
            import requests
            body = {"topic": topic, "payload": payload}
            requests.post(self._endpoint, json=body, headers=self._headers, timeout=self._timeout)
        except Exception:
            # best-effort: ignore failures in publisher
            pass
