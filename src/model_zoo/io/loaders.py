from __future__ import annotations
"""Simple DataRef loader utilities.

DefaultDataLoader resolves INLINE refs to their Python value and returns URI
refs unchanged by default. Applications can extend this to read from cloud
storage, local files, or databases.
"""
from typing import Any
from ..contracts.io import DataRef, DataRefType


class DefaultDataLoader:
    def load(self, ref: DataRef) -> Any:
        if ref.ref_type == DataRefType.INLINE:
            return ref.inline
        # For URIs we return the DataRef for the callable to handle. Users can
        # subclass and implement scheme-specific fetching.
        return ref

