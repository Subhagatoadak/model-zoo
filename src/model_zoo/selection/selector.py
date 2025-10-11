"""Simple model selection service for agents.

This is a minimal baseline that filters by kind/tags and optional metadata
constraints, then returns a deterministic choice (latest version by string).
Applications can replace this with a more sophisticated selector.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from ..registry.models import ModelRegistry, ModelSpec, ModelKind


@dataclass
class SelectionCriteria:
    """Filter fields used to select a model from the registry."""
    kind: Optional[ModelKind] = None
    required_tags: List[str] = None
    metadata_equals: Dict[str, Any] = None


class ModelSelectionService:
    def __init__(self, registry: ModelRegistry) -> None:
        """Create a selector backed by a ModelRegistry."""
        self._registry = registry

    def select(self, criteria: SelectionCriteria) -> Optional[ModelSpec]:
        """Return a single best match or None if no candidates.

        Policy: filter by kind/tags/metadata, then choose max(version) by string.
        """
        required_tags = criteria.required_tags or []
        meta_eq = criteria.metadata_equals or {}
        candidates = []
        for spec in self._registry.list():
            if criteria.kind and spec.kind != criteria.kind:
                continue
            if any(t not in spec.tags for t in required_tags):
                continue
            if any(spec.metadata.get(k) != v for k, v in meta_eq.items()):
                continue
            candidates.append(spec)
        if not candidates:
            return None
        # naive string compare; applications can implement semver if needed
        return max(candidates, key=lambda s: s.version)
