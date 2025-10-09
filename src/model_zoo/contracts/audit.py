from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

@dataclass
class AuditEvent:
    kind: str
    who: str
    when: str
    ref_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)

class AuditLog(ABC):
    @abstractmethod
    def append(self, event: AuditEvent) -> None: ...
