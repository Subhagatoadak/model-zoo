from __future__ import annotations
"""Example metric callables for in‑process evaluation via LibraryMetricRunner."""
from typing import Iterable, Optional, Dict, Any


def _to_list(x: Optional[Iterable[float]]):
    if x is None:
        return []
    return list(x)


def r2_score(y_true, y_pred=None, proba=None, samples=None, params=None, extra=None) -> float:
    yt = _to_list(y_true)
    yp = _to_list(y_pred)
    if not yt or not yp or len(yt) != len(yp):
        raise ValueError("r2_score: y_true and y_pred must be same non-zero length")
    mean_y = sum(yt) / len(yt)
    ss_tot = sum((yi - mean_y) ** 2 for yi in yt)
    ss_res = sum((yi - yhat) ** 2 for yi, yhat in zip(yt, yp))
    return 1.0 - (ss_res / (ss_tot + 1e-12)) if ss_tot > 0 else 0.0


def mse(y_true, y_pred=None, proba=None, samples=None, params=None, extra=None) -> float:
    yt = _to_list(y_true)
    yp = _to_list(y_pred)
    if not yt or not yp or len(yt) != len(yp):
        raise ValueError("mse: y_true and y_pred must be same non-zero length")
    err = [(yi - yhat) ** 2 for yi, yhat in zip(yt, yp)]
    return sum(err) / len(err)


def mae(y_true, y_pred=None, proba=None, samples=None, params=None, extra=None) -> float:
    yt = _to_list(y_true)
    yp = _to_list(y_pred)
    if not yt or not yp or len(yt) != len(yp):
        raise ValueError("mae: y_true and y_pred must be same non-zero length")
    err = [abs(yi - yhat) for yi, yhat in zip(yt, yp)]
    return sum(err) / len(err)
