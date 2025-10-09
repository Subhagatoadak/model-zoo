from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum

class DataRefType(Enum):
    INLINE = "inline"
    URI = "uri"

@dataclass
class DataRef:
    ref_type: DataRefType
    uri: Optional[str] = None
    inline: Optional[Any] = None
    format: Optional[str] = None
    schema_hint: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"ref_type": self.ref_type.value, "uri": self.uri, "inline": self.inline, "format": self.format, "schema_hint": self.schema_hint}

@dataclass
class SecretRef:
    provider: str
    name: str
    version: Optional[str] = None
    scope: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionContext:
    random_seed: Optional[int] = None
    timeout_s: Optional[int] = None
    device: Optional[str] = None
    resources: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, Any] = field(default_factory=dict)
