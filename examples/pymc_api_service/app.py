from __future__ import annotations

import math
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pymc as pm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class RunPayload(BaseModel):
    model_id: str = Field(..., description="Logical model identifier")
    inputs: Dict[str, List[float]]
    params: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    evaluation: Optional[Dict[str, Any]] = None


app = FastAPI(title="Dummy PyMC Model API", version="0.1.0")


def _map_linear_regression(x_values: List[float], y_values: List[float]) -> tuple[list[float], Dict[str, float]]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    if x.shape != y.shape:
        raise ValueError("X and y must have the same length")

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=10.0)
        beta = pm.Normal("beta", mu=0.0, sigma=10.0)
        sigma = pm.Exponential("sigma", lam=1.0)
        mu = alpha + beta * x
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
        map_est = pm.find_MAP(progressbar=False)

    alpha_map = float(map_est.get("alpha", 0.0))
    beta_map = float(map_est.get("beta", 0.0))
    sigma_map = float(map_est.get("sigma", map_est.get("sigma_log__", 0.0)))
    if "sigma_log__" in map_est:
        sigma_map = float(math.exp(map_est["sigma_log__"]))

    preds = alpha_map + beta_map * x
    diagnostics = {
        "alpha_map": alpha_map,
        "beta_map": beta_map,
        "sigma_map": sigma_map,
    }

    return preds.tolist(), diagnostics


@app.post("/runs")
async def run_model(payload: RunPayload) -> Dict[str, Any]:
    x_values = payload.inputs.get("X")
    y_values = payload.inputs.get("y")
    if x_values is None or y_values is None:
        raise HTTPException(status_code=400, detail="Inputs must include 'X' and 'y'")

    try:
        predictions, diagnostics = _map_linear_regression(x_values, y_values)
    except Exception as exc:  # pragma: no cover - PyMC errors bubble to client
        raise HTTPException(status_code=500, detail=f"PyMC execution failed: {exc}") from exc

    y = np.asarray(y_values, dtype=float)
    pred_arr = np.asarray(predictions, dtype=float)
    residuals = y - pred_arr
    ss_res = float(np.sum(np.square(residuals)))
    ss_tot = float(np.sum(np.square(y - y.mean()))) if y.size else 0.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0

    summary = {
        "Call": {"model_id": payload.model_id, "params": payload.params},
        "Coefficients": {
            "alpha_map": diagnostics["alpha_map"],
            "beta_map": diagnostics["beta_map"],
        },
        "Fit": {
            "r2": r2,
            "sigma_map": diagnostics["sigma_map"],
        },
        "Residuals": {
            "rss": ss_res,
        },
    }

    return {
        "run_id": str(uuid.uuid4()),
        "status": "succeeded",
        "predictions": predictions,
        "diagnostics": diagnostics,
        "summary": summary,
    }
