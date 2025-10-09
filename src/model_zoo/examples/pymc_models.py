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

    # Normalize X to (n, p), y to (n,)
    if isinstance(X[0], dict):
        feature_names = params.get("feature_names")
        keys = list(X[0].keys())
        if feature_names is None:
            feature_names = keys
        else:
            missing = [k for k in feature_names if k not in keys]
            if missing:
                raise ValueError(f"feature_names not found in X dict keys: {missing}")
            keys = feature_names
        X_mat = [[float(row[k]) for k in keys] for row in X]
        X_arr = np.array(X_mat, dtype=float)
    elif isinstance(X[0], (int, float)):
        X_arr = np.array(X, dtype=float).reshape(-1, 1)
        feature_names = [params.get("feature_name", "x")]
    else:
        X_arr = np.array(X, dtype=float)
        feature_names = params.get("feature_names") or [f"x{i}" for i in range(X_arr.shape[1])]
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
    chains = int(params.get("chains", 1))
    cores = int(params.get("cores", 1))
    progressbar = bool(params.get("progressbar", False))
    compute_convergence_checks = bool(params.get("compute_convergence_checks", False))

    priors = params.get("priors", {}) or {}
    beta_prior = priors.get("beta", {})
    p = X_arr.shape[1]
    beta_dist_raw = beta_prior.get("distribution", "normal")
    # Helpers to extract possibly vectorized params
    def _get_param(name: str, default: float, i: int) -> float:
        val = beta_prior.get(name, None)
        if isinstance(val, list):
            if len(val) != p:
                raise ValueError(f"beta.{name} length must equal number of features")
            return float(val[i]) if val[i] is not None else float(default)
        if val is None:
            return float(default)
        return float(val)
    sigma_prior = priors.get("sigma", {})
    sigma_dist = (sigma_prior.get("distribution") or "half_normal").lower()
    sigma_scale = float(sigma_prior.get("scale", 1.0))

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=10.0)
        # Build per-feature beta priors supporting different distributions
        beta_vars = []
        for i, name in enumerate(feature_names):
            # distribution can be scalar or list; allow per-feature
            if isinstance(beta_dist_raw, list):
                if len(beta_dist_raw) != p:
                    raise ValueError("beta.distribution length must equal number of features")
                dist_i = str(beta_dist_raw[i]).lower()
            else:
                dist_i = str(beta_dist_raw).lower()
            # sanitize name for PyMC RV
            rv_name = f"beta_{name}".replace(" ", "_").replace("[", "_").replace("]", "_")
            if dist_i in ("normal",):
                mu_i = _get_param("mu", 0.0, i)
                tau_i = _get_param("tau", 10.0, i)
                beta_i = pm.Normal(rv_name, mu=mu_i, sigma=tau_i)
            elif dist_i in ("half_normal", "halfnormal"):
                scale_i = _get_param("scale", 1.0, i)
                beta_i = pm.HalfNormal(rv_name, sigma=scale_i)
            elif dist_i in ("gamma", "erlang"):
                shape_i = _get_param("shape", 2.0, i)
                rate_i = _get_param("rate", 1.0, i)
                beta_i = pm.Gamma(rv_name, alpha=shape_i, beta=rate_i)
            else:
                raise ValueError(f"Unsupported beta prior distribution: {dist_i}")
            beta_vars.append(beta_i)

        beta_vec = pm.math.stack(beta_vars)

        # sigma prior
        if sigma_dist in ("half_normal", "halfnormal"):
            sigma = pm.HalfNormal("sigma", sigma=sigma_scale)
        elif sigma_dist in ("gamma", "erlang"):
            # shape-rate parameterization
            shape = float(sigma_prior.get("shape") or sigma_prior.get("alpha") or 2.0)
            rate = float(sigma_prior.get("rate") or sigma_prior.get("beta") or max(1e-6, 1.0))
            sigma = pm.Gamma("sigma", alpha=shape, beta=rate)
        else:
            raise ValueError(f"Unsupported sigma prior distribution: {sigma_dist}")
        mu = alpha + pm.math.dot(X_train, beta_vec)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=getattr(context, "random_seed", None),
            progressbar=progressbar,
            compute_convergence_checks=compute_convergence_checks,
        )

    # Posterior means
    import numpy as _np
    alpha_mean = float(idata.posterior["alpha"].values.mean())
    # Extract per-feature posterior means by variable name
    beta_means: List[float] = []
    for name in feature_names:
        rv_name = f"beta_{name}".replace(" ", "_").replace("[", "_").replace("]", "_")
        beta_means.append(float(idata.posterior[rv_name].values.mean()))

    beta_mean_vec = _np.array(beta_means)
    y_pred = (alpha_mean + X_arr.dot(beta_mean_vec)).tolist()

    diagnostics: Dict[str, Any] = {
        "alpha": alpha_mean,
        "draws": draws,
        "tune": tune,
        "target_accept": target_accept,
        "chains": chains,
    }
    # add named coefficients
    for i, name in enumerate(feature_names):
        diagnostics[f"coef[{name}]"] = float(beta_mean_vec[i])

    return {
        "predictions": y_pred,
        "diagnostics": diagnostics,
        "metadata": {
            "model": "pymc_linear_nuts",
            "feature_names": feature_names,
            "priors": {"beta": {"distribution": beta_dist_raw}, "sigma": {"distribution": sigma_dist}},
        },
    }
