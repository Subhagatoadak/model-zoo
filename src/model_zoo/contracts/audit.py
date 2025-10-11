"""Audit event contracts for recording important lifecycle events."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

@dataclass
class AuditEvent:
    """Structured audit event used by the `AuditLog` interface."""
    kind: str
    who: str
    when: str
    ref_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)

class AuditLog(ABC):
    """Abstract audit sink (append‑only)."""
    @abstractmethod
    def append(self, event: AuditEvent) -> None:
        """Persist an audit event (best‑effort)."""
        ...
