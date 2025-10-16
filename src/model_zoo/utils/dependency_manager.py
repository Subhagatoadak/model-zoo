"""Utilities for resolving and caching model dependency packages."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, cast
from urllib.parse import quote, urlsplit, urlunsplit

from packaging.requirements import Requirement

logger = logging.getLogger(__name__)


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


def _sanitize_url(url: str) -> str:
    parsed = urlsplit(url)
    netloc = parsed.netloc
    if "@" in netloc:
        creds, host = netloc.split("@", 1)
        if ":" in creds:
            user, _ = creds.split(":", 1)
            creds = f"{user}:<redacted>"
        else:
            creds = "<redacted>"
        netloc = f"{creds}@{host}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


@dataclass
class RepositoryCredentials:
    """Connection information for a Python package repository."""

    url: str
    username: Optional[str] = None
    password: Optional[str] = None
    password_env: Optional[str] = None

    def _resolved_password(self) -> Optional[str]:
        if self.password is not None:
            return self.password
        if self.password_env:
            return os.environ.get(self.password_env)
        return None

    def resolved_url(self) -> str:
        """Return the repository URL including credentials if provided."""
        raw_url = str(self.url or "").strip()
        if not raw_url:
            raise DependencyResolutionError("Repository URL must be a non-empty string")

        parsed = urlsplit(raw_url)
        if not parsed.scheme or not parsed.netloc:
            raise DependencyResolutionError(f"Repository URL '{raw_url}' must include a scheme and hostname")

        if parsed.username or parsed.password:
            return raw_url

        password = self._resolved_password()
        if (self.username or password) and parsed.hostname:
            if password is None:
                raise DependencyResolutionError(
                    f"Repository credentials require password via explicit value or environment '{self.password_env}'"
                )
            user = quote(str(self.username or ""), safe="")
            pwd = quote(str(password), safe="")
            host = parsed.hostname
            if parsed.port:
                host = f"{host}:{parsed.port}"
            netloc = f"{user}:{pwd}@{host}"
            return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))

        if self.username and password is None:
            raise DependencyResolutionError(
                "Repository credentials provided a username without a password; set 'password' or 'password_env'"
            )

        return raw_url

    def secret_tokens(self) -> List[str]:
        password = self._resolved_password()
        if not password:
            return []
        tokens = [password]
        tokens.append(quote(password, safe=""))
        return tokens


@dataclass
class DependencyOptions:
    """Optional pip flags that influence dependency installation."""

    index: Optional[RepositoryCredentials] = None
    extras: List[RepositoryCredentials] = field(default_factory=list)
    trusted_hosts: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)

    def apply_env(self) -> None:
        for key, value in self.env.items():
            if value is None:
                continue
            os.environ.setdefault(str(key), str(value))

    def pip_args(self, include_meta: bool = False) -> List[str] | Tuple[List[str], List[str], List[str]]:
        args: List[str] = []
        sanitized: List[str] = []
        secrets: List[str] = []

        def add(option: str, value: str, safe_value: str) -> None:
            args.extend([option, value])
            sanitized.extend([option, safe_value])

        if self.index is not None:
            url = self.index.resolved_url()
            add("--index-url", url, _sanitize_url(url))
            secrets.extend(self.index.secret_tokens())
        for repo in self.extras:
            url = repo.resolved_url()
            add("--extra-index-url", url, _sanitize_url(url))
            secrets.extend(repo.secret_tokens())
        for host in self.trusted_hosts:
            host_clean = str(host or "").strip()
            if host_clean:
                add("--trusted-host", host_clean, host_clean)

        if include_meta:
            return args, sanitized, [s for s in secrets if s]
        return args


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

    def ensure(
        self,
        dependencies: Iterable[str],
        options: Optional[DependencyOptions] = None,
    ) -> Optional[DependencyEnvironment]:
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

            base_cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-warn-script-location",
                "--target",
                str(target),
            ]
            sanitized_cmd = list(base_cmd)
            secret_tokens: List[str] = []

            if options is not None:
                options.apply_env()
                pip_args_meta = options.pip_args(include_meta=True)
                pip_args, pip_args_sanitized, secrets = cast(Tuple[List[str], List[str], List[str]], pip_args_meta)
                base_cmd.extend(pip_args)
                sanitized_cmd.extend(pip_args_sanitized)
                secret_tokens.extend(secrets)

            base_cmd.extend(normalized)
            sanitized_cmd.extend(normalized)

            logger.info(
                "Installing dependencies via pip",
                extra={"target": str(target), "requirements": normalized, "command": sanitized_cmd},
            )

            proc = subprocess.run(base_cmd, capture_output=True, text=True, check=False)
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""

            for secret in secret_tokens:
                stdout = stdout.replace(secret, "<redacted>")
                stderr = stderr.replace(secret, "<redacted>")

            log_payload = {
                "command": sanitized_cmd,
                "returncode": proc.returncode,
                "stdout": stdout.strip(),
                "stderr": stderr.strip(),
            }
            (target / "pip_install.json").write_text(json.dumps(log_payload, indent=2), encoding="utf-8")

            if proc.returncode != 0:
                raise DependencyResolutionError(
                    "Failed to install dependencies (see pip_install.json for details): "
                    f"{stderr.strip() or stdout.strip()}"
                )

            marker.write_text(json.dumps({"requirements": normalized}), encoding="utf-8")

            logger.info(
                "Dependency installation completed",
                extra={"target": str(target), "requirements": normalized},
            )

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
    "DependencyOptions",
    "RepositoryCredentials",
]
