from __future__ import annotations
import logging
import os
from typing import Optional


def configure_logging(level: Optional[str] = None) -> None:
    """Configure root logging with a sensible default formatter.

    Respects env LOG_LEVEL if provided. Safe to call multiple times.
    """
    lvl = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    if isinstance(lvl, str):
        try:
            numeric_level = getattr(logging, lvl, logging.INFO)
        except Exception:
            numeric_level = logging.INFO
    else:
        numeric_level = logging.INFO

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    else:
        logging.getLogger().setLevel(numeric_level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

