from __future__ import annotations
from typing import Any, Dict, List


def pymc_linear_nuts(inputs: Dict[str, Any], params: Dict[str, Any], context) -> Dict[str, Any]:
    """Bayesian linear regression via PyMC NUTS sampler.

    Requires pymc, numpy, and arviz. Handles single-feature X.
    params: draws, tune, target_accept, priors (beta normal; sigma half-normal or gamma on precision unsupported here).
    """
    try:
        import numpy as np
        import pymc as pm
    except Exception as e:
        raise RuntimeError("pymc and numpy are required for pymc_linear_nuts") from e

    X = inputs.get("X")
    y = inputs.get("y")
    if X is None or y is None:
        raise ValueError("inputs must contain 'X' and 'y'")

    X_arr = np.array(X, dtype=float).reshape(-1)
    y_arr = np.array(y, dtype=float).reshape(-1)

    # Optional: train subset
    train_idx = params.get("train_indices")
    if train_idx is not None:
        X_train = X_arr[train_idx]
        y_train = y_arr[train_idx]
    else:
        X_train, y_train = X_arr, y_arr

    draws = int(params.get("draws", 1000))
    tune = int(params.get("tune", 1000))
    target_accept = float(params.get("target_accept", 0.9))

    priors = params.get("priors", {}) or {}
    beta_prior = priors.get("beta", {})
    beta_mu = float(beta_prior.get("mu", 0.0))
    beta_tau = float(beta_prior.get("tau", 10.0))
    sigma_prior = priors.get("sigma", {})
    sigma_scale = float(sigma_prior.get("scale", 1.0))

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=10.0)
        beta = pm.Normal("beta", mu=beta_mu, sigma=beta_tau)
        sigma = pm.HalfNormal("sigma", sigma=sigma_scale)
        mu = alpha + beta * X_train
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=1,
            target_accept=target_accept,
            random_seed=getattr(context, "random_seed", None),
            progressbar=False,
        )

    # Posterior means
    try:
        import numpy as np
        alpha_mean = float(idata.posterior["alpha"].values.mean())
        beta_mean = float(idata.posterior["beta"].values.mean())
    except Exception:
        # Fallback: use last draw
        alpha_mean = float(idata.posterior["alpha"].values.reshape(-1)[-1])
        beta_mean = float(idata.posterior["beta"].values.reshape(-1)[-1])

    y_pred = (alpha_mean + beta_mean * X_arr).tolist()

    diagnostics: Dict[str, Any] = {
        "alpha": alpha_mean,
        "beta": beta_mean,
        "draws": draws,
        "tune": tune,
        "target_accept": target_accept,
    }

    return {
        "predictions": y_pred,
        "diagnostics": diagnostics,
        "metadata": {"model": "pymc_linear_nuts"},
    }

