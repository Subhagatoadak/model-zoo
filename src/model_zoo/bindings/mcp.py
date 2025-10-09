from __future__ import annotations
"""Options for invoking models via Model Context Protocol (MCP)."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class MCPBindingOptions:
    """Connection + invocation options for MCP bindings.

    Depending on the transport, either `command` (stdio) or `endpoint`
    (streamable_http) is required. Additional metadata such as headers, tool
    name, and protocol version can be provided.
    """
    transport: str                     # "stdio" | "streamable_http"
    endpoint: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    protocol_version: str = "2025-06-18"
    headers: Dict[str, str] = field(default_factory=dict)
    tool_name: str = "models.run"
    prompt_name: Optional[str] = None
    resource_map: Dict[str, str] = field(default_factory=dict)
    session_resume: bool = True
    timeout_s: int = 3600
