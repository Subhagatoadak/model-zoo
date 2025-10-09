from __future__ import annotations
"""Run store interfaces and an in-memory implementation."""
from dataclasses import dataclass
from typing import Dict, Optional
from ..contracts.run_requests import ModelRunResult


@dataclass
class RunRecord:
    run_id: str
    result: ModelRunResult


class RunStore:
    def save(self, rec: RunRecord) -> None:
        raise NotImplementedError

    def get(self, run_id: str) -> Optional[RunRecord]:
        raise NotImplementedError


class InMemoryRunStore(RunStore):
    def __init__(self) -> None:
        self._recs: Dict[str, RunRecord] = {}

    def save(self, rec: RunRecord) -> None:
        self._recs[rec.run_id] = rec

    def get(self, run_id: str) -> Optional[RunRecord]:
        return self._recs.get(run_id)

