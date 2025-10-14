"""Helpers for building model input variables from configuration."""
from __future__ import annotations

from typing import Any, Dict, Tuple

from .config_loader import VariableConfig, TransformSpec
from .transforms import apply_transforms, merge_transform_registries


def _specs_to_payload(specs: list[TransformSpec]) -> list[dict[str, Any]]:
    return [{"id": spec.transform_id, "params": spec.params} for spec in specs]


def build_variables(
    config: VariableConfig,
    data_sources: Dict[str, Any],
    custom_transforms: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return processed variables and grouped payloads.

    Args:
        config: variable configuration definitions and groups.
        data_sources: mapping of source name -> raw values.
        custom_transforms: optional registry extensions.

    Returns:
        A tuple of (variables, group_outputs) where each dictionary contains the
        processed values ready to be sent to model runners.
    """

    registry = merge_transform_registries(custom_transforms)
    resolved: Dict[str, Any] = {}
    for var in config.active_variables():
        source_name = var.source
        if source_name in data_sources:
            raw_value = data_sources[source_name]
        elif source_name in resolved:
            raw_value = resolved[source_name]
        else:
            raise KeyError(f"Missing data source for variable '{source_name}'")
        transformed = apply_transforms(raw_value, _specs_to_payload(var.transforms), registry)
        resolved[var.name] = transformed

    grouped: Dict[str, Any] = {}
    for group in config.groups:
        items = [resolved.get(name) for name in group.variables]
        if not all(item is not None for item in items):
            missing = [name for name, item in zip(group.variables, items) if item is None]
            raise KeyError(f"Variable group '{group.group_id}' missing variables: {missing}")
        if len(items) == 1:
            base_value = items[0]
        else:
            base_value = {name: resolved[name] for name in group.variables}
        transformed = apply_transforms(base_value, _specs_to_payload(group.transforms), registry)
        key = group.output_name or group.group_id
        grouped[key] = transformed

    merged = dict(resolved)
    merged.update(grouped)
    return merged, grouped
