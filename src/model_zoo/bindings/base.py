from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum

class BindingKind(Enum):
    API = "api"
    LIBRARY = "library"
    PIPELINE = "pipeline"
    MCP = "mcp"

@dataclass
class ExecutionBinding:
    bound_id: str
    kind: BindingKind
    endpoint: Optional[str] = None
    method: Optional[str] = None
    module: Optional[str] = None
    callable: Optional[str] = None
    pipeline_id: Optional[str] = None
    orchestrator: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)
    mcp: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bound_id": self.bound_id, "kind": self.kind.value,
            "endpoint": self.endpoint, "method": self.method,
            "module": self.module, "callable": self.callable,
            "pipeline_id": self.pipeline_id, "orchestrator": self.orchestrator,
            "options": self.options, "mcp": None if self.mcp is None else self.mcp.__dict__,
        }
