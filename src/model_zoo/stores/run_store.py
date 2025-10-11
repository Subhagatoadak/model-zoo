"""Run store interfaces and an in-memory implementation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
from ..contracts.run_requests import ModelRunResult
from threading import Lock


@dataclass
class RunRecord:
    """A stored run record containing the latest `ModelRunResult`."""
    run_id: str
    result: ModelRunResult


class RunStore:
    """Abstract interface for persisting run records."""

    def save(self, rec: RunRecord) -> None:
        """Persist or update a run record."""
        raise NotImplementedError

    def get(self, run_id: str) -> Optional[RunRecord]:
        """Retrieve a run record by id, or None if absent."""
        raise NotImplementedError


class InMemoryRunStore(RunStore):
    def __init__(self) -> None:
        self._recs: Dict[str, RunRecord] = {}
        self._lock = Lock()

    def save(self, rec: RunRecord) -> None:
        """Store the run record in memory, replacing any existing one."""
        with self._lock:
            self._recs[rec.run_id] = rec

    def get(self, run_id: str) -> Optional[RunRecord]:
        """Return a previously saved RunRecord, if present."""
        with self._lock:
            return self._recs.get(run_id)
