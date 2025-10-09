from __future__ import annotations
"""Hyperparameter schema primitives used by registries and validators."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

class DType(Enum):
    """JSON‑schema compatible base types used in HyperParamField."""
    INT = "int"
    FLOAT = "number"
    BOOL = "boolean"
    STR = "string"
    CATEGORICAL = "string"
    DICT = "object"
    LIST = "array"

@dataclass
class HyperParamField:
    """A single hyperparameter definition with validation hints."""
    name: str
    dtype: DType
    required: bool = False
    default: Optional[Any] = None
    description: str = ""
    choices: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    tunable: bool = True
    group: Optional[str] = None
    example: Optional[Any] = None

    def to_json_schema(self) -> Dict[str, Any]:
        """Serialize field into a minimal JSON Schema fragment."""
        schema: Dict[str, Any] = {
            "title": self.name, "description": self.description, "type": self.dtype.value
        }
        if self.choices is not None: schema["enum"] = self.choices
        if self.min_value is not None: schema["minimum"] = self.min_value
        if self.max_value is not None: schema["maximum"] = self.max_value
        if self.pattern is not None: schema["pattern"] = self.pattern
        if self.default is not None: schema["default"] = self.default
        if self.example is not None: schema["examples"] = [self.example]
        return schema

@dataclass
class HyperParamSpace:
    """A collection of hyperparameters (name -> HyperParamField)."""
    fields: Dict[str, HyperParamField] = field(default_factory=dict)

    def to_json_schema(self) -> Dict[str, Any]:
        """Serialize the space into a minimal JSON Schema object."""
        properties = {n: f.to_json_schema() for n, f in self.fields.items()}
        required = [n for n, f in self.fields.items() if f.required]
        return {"type": "object", "properties": properties, "required": required, "additionalProperties": True}
