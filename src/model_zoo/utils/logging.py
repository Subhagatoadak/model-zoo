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


def quiet_third_party_loggers(level: Optional[str] = None) -> None:
    """Lower log level for noisy third-party libraries (arviz, pymc).

    By default sets them to WARNING. You can override via argument or env
    MZOO_QUIET_TPL (truthy) to enable, and MZOO_TPL_LEVEL to set level.
    """
    enable = os.getenv("MZOO_QUIET_TPL", "true").lower() in ("1", "true", "yes")
    if not enable:
        return
    lvl = (level or os.getenv("MZOO_TPL_LEVEL") or "WARNING").upper()
    try:
        numeric_level = getattr(logging, lvl, logging.WARNING)
    except Exception:
        numeric_level = logging.WARNING
    for name in [
        "arviz", "arviz.preview", "arviz.stats", "arviz.plots",
        "pymc", "pymc.sampling.mcmc", "pymc.stats.convergence",
    ]:
        logging.getLogger(name).setLevel(numeric_level)
