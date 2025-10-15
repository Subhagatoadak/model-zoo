"""Utilities for resolving and caching model dependency packages."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Iterable, List, Optional, Sequence

from packaging.requirements import Requirement


class DependencyResolutionError(RuntimeError):
    """Raised when dependency preparation fails."""


@dataclass
class DependencyEnvironment:
    """Represents a prepared dependency environment."""

    key: str
    location: Path
    requirements: Sequence[str]

    @property
    def import_paths(self) -> List[str]:
        """Return filesystem paths that should be added to ``sys.path``."""
        return [str(self.location)]


class DependencyManager:
    """Prepare and cache pip-style dependencies for dynamic model bindings."""

    def __init__(self, base_dir: Optional[Path | str] = None) -> None:
        root = Path(base_dir) if base_dir is not None else Path.home() / ".model_zoo" / "dependencies"
        self._base_dir = root.resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def normalize(self, dependencies: Iterable[str]) -> List[str]:
        """Normalize and validate dependency strings."""
        seen = set()
        normalized: List[str] = []
        for dep in dependencies:
            if dep is None:
                continue
            dep = str(dep).strip()
            if not dep:
                continue
            try:
                # Validate using packaging; keeps original string for reproducibility
                Requirement(dep)
            except Exception as exc:  # pragma: no cover - packaging gives detailed message
                raise DependencyResolutionError(f"Invalid dependency specification '{dep}': {exc}") from exc
            if dep not in seen:
                seen.add(dep)
                normalized.append(dep)
        return normalized

    def ensure(self, dependencies: Iterable[str]) -> Optional[DependencyEnvironment]:
        """Ensure the given dependencies are installed into a cached target directory."""
        normalized = self.normalize(dependencies)
        if not normalized:
            return None

        joined = "\n".join(normalized)
        key = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
        target = self._base_dir / key
        marker = target / ".installed"

        if marker.exists():
            return DependencyEnvironment(key=key, location=target, requirements=tuple(normalized))

        with self._lock:
            if marker.exists():
                return DependencyEnvironment(key=key, location=target, requirements=tuple(normalized))
            target.mkdir(parents=True, exist_ok=True)

            if os.environ.get("MODEL_ZOO_SKIP_DEP_INSTALL") == "1":
                marker.write_text(json.dumps({"requirements": normalized}), encoding="utf-8")
                return DependencyEnvironment(key=key, location=target, requirements=tuple(normalized))

            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-warn-script-location",
                "--target",
                str(target),
                *normalized,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                raise DependencyResolutionError(
                    "Failed to install dependencies: "
                    f"{proc.stdout.strip()}\n{proc.stderr.strip()}"
                )

            marker.write_text(json.dumps({"requirements": normalized}), encoding="utf-8")

        return DependencyEnvironment(key=key, location=target, requirements=tuple(normalized))

    def apply(self, env: DependencyEnvironment) -> None:
        """Add the dependency environment to ``sys.path`` for import resolution."""
        for path in env.import_paths:
            if path not in sys.path:
                sys.path.insert(0, path)


__all__ = [
    "DependencyManager",
    "DependencyEnvironment",
    "DependencyResolutionError",
]
