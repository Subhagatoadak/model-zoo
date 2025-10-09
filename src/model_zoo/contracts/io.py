from __future__ import annotations
"""I/O data contracts used by run requests and results."""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum

class DataRefType(Enum):
    """The storage mechanism for a piece of data.

    - INLINE: value is embedded in the request/response as a Python object.
    - URI: value must be dereferenced from a URI by the runner/binding.
    """

    INLINE = "inline"
    URI = "uri"

@dataclass
class DataRef:
    """A typed reference to data used by models/metrics.

    Either ``inline`` or ``uri`` should be set depending on ``ref_type``. The
    ``format`` and ``schema_hint`` fields are advisory and may be used by
    runners/loaders.
    """
    ref_type: DataRefType
    uri: Optional[str] = None
    inline: Optional[Any] = None
    format: Optional[str] = None
    schema_hint: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON‑serializable dictionary."""
        return {"ref_type": self.ref_type.value, "uri": self.uri, "inline": self.inline, "format": self.format, "schema_hint": self.schema_hint}

@dataclass
class SecretRef:
    """A reference to a secret managed by an external provider."""
    provider: str
    name: str
    version: Optional[str] = None
    scope: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionContext:
    """Optional execution hints for runners (timeouts, device, resources)."""
    random_seed: Optional[int] = None
    timeout_s: Optional[int] = None
    device: Optional[str] = None
    resources: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, Any] = field(default_factory=dict)
