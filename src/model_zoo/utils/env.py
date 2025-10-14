"""Environment file helpers for examples and runners."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
from urllib.parse import quote_plus


def read_env_file(path: str | Path) -> Dict[str, str]:
    """Parse a simple KEY=VALUE env file into a dictionary."""

    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Env file not found: {resolved}")

    values: Dict[str, str] = {}
    for raw_line in resolved.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def build_postgres_dsn(settings: Dict[str, str]) -> Tuple[str, str]:
    """Compose a PostgreSQL DSN and table name from env settings."""

    user = settings.get("POSTGRES_USER") or settings.get("POSTGRES_USERNAME")
    password = settings.get("POSTGRES_PASSWORD")
    host = settings.get("POSTGRES_HOST", "localhost")
    port = settings.get("POSTGRES_PORT", "5432")
    database = settings.get("POSTGRES_DB") or settings.get("POSTGRES_DATABASE")
    table = settings.get("POSTGRES_TABLE", "model_runs")

    if not user or not password or not database:
        required = [("POSTGRES_USER", user), ("POSTGRES_PASSWORD", password), ("POSTGRES_DB", database)]
        missing = [name for name, value in required if not value]
        raise ValueError(f"Missing required PostgreSQL settings: {missing}")

    quoted_user = quote_plus(user)
    quoted_password = quote_plus(password)
    dsn = f"postgresql://{quoted_user}:{quoted_password}@{host}:{port}/{database}"
    return dsn, table

