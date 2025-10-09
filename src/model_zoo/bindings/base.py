from __future__ import annotations
"""Binding primitives that describe how to execute models and metrics.

Bindings are kept separate from registries and attached at request time (or
resolved via a BindingRegistry). They encode transport details such as
module/callable for in‑process execution, HTTP endpoints, pipeline IDs, or MCP
options.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum

class BindingKind(Enum):
    """Mechanism by which a binding executes."""
    API = "api"
    LIBRARY = "library"
    PIPELINE = "pipeline"
    MCP = "mcp"

@dataclass
class ExecutionBinding:
    """Concrete description of how to execute a model/metric.

    Fields are optional depending on `kind`. For example, LIBRARY bindings use
    `module` + `callable`; API bindings use `endpoint` + `method`; MCP bindings
    attach options describing transport and tool invocation.
    """
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
        """Serialize to a JSON‑serializable dictionary for logs/APIs."""
        return {
            "bound_id": self.bound_id, "kind": self.kind.value,
            "endpoint": self.endpoint, "method": self.method,
            "module": self.module, "callable": self.callable,
            "pipeline_id": self.pipeline_id, "orchestrator": self.orchestrator,
            "options": self.options, "mcp": None if self.mcp is None else self.mcp.__dict__,
        }
