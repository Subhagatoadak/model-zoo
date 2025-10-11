"""Simple DataRef loader utilities.

DefaultDataLoader resolves INLINE refs to their Python value and returns URI
refs unchanged by default. Applications can extend this to read from cloud
storage, local files, or databases.
"""
from __future__ import annotations
from typing import Any
from ..contracts.io import DataRef, DataRefType


class DefaultDataLoader:
    """Resolve a `DataRef` into a Python value.

    INLINE refs return their `inline` value directly. URI refs are returned
    unchanged by default; downstream callables may decide how to load them, or
    you may subclass this loader to handle specific schemes (e.g., file/s3).
    """

    def load(self, ref: DataRef) -> Any:
        """Load a `DataRef` into a usable value for in‑process callables."""
        if ref.ref_type == DataRefType.INLINE:
            return ref.inline
        # For URIs we return the DataRef for the callable to handle. Users can
        # subclass and implement scheme-specific fetching.
        return ref
