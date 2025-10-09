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
    beta_dist = (beta_prior.get("distribution") or "normal").lower()
    # Support scalar or per-feature vectors for mu and tau
    p = X_arr.shape[1]
    mu_raw = beta_prior.get("mu", 0.0)
    tau_raw = beta_prior.get("tau", 10.0)
    if isinstance(mu_raw, list):
        if len(mu_raw) != p: raise ValueError("beta.mu length must equal number of features")
        beta_mu = np.array(mu_raw, dtype=float)
    else:
        beta_mu = float(mu_raw)
    if isinstance(tau_raw, list):
        if len(tau_raw) != p: raise ValueError("beta.tau length must equal number of features")
        beta_tau = np.array(tau_raw, dtype=float)
    else:
        beta_tau = float(tau_raw)
    sigma_prior = priors.get("sigma", {})
    sigma_dist = (sigma_prior.get("distribution") or "half_normal").lower()
    sigma_scale = float(sigma_prior.get("scale", 1.0))

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=10.0)
        # beta prior
        if beta_dist not in ("normal",):
            raise ValueError(f"Unsupported beta prior distribution: {beta_dist}")
        if p == 1:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_tau)
            mu = alpha + beta * X_train
        else:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_tau, shape=p)
            mu = alpha + pm.math.dot(X_train, beta)

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
    beta_vals = idata.posterior["beta"].values
    beta_mean_arr = _np.array(beta_vals).mean(axis=(0, 1))  # average over chain, draw
    beta_mean = float(beta_mean_arr) if p == 1 else beta_mean_arr.reshape(-1)

    y_pred = (alpha_mean + (X_arr * beta_mean if p == 1 else X_arr.dot(beta_mean))).tolist()

    diagnostics: Dict[str, Any] = {
        "alpha": alpha_mean,
        "draws": draws,
        "tune": tune,
        "target_accept": target_accept,
        "chains": chains,
    }
    # add named coefficients
    if p == 1:
        name = feature_names[0]
        diagnostics[f"coef[{name}]"] = float(beta_mean)
    else:
        for i, name in enumerate(feature_names):
            diagnostics[f"coef[{name}]"] = float(beta_mean[i])

    return {
        "predictions": y_pred,
        "diagnostics": diagnostics,
        "metadata": {
            "model": "pymc_linear_nuts",
            "feature_names": feature_names,
            "priors": {"beta": {"distribution": beta_dist}, "sigma": {"distribution": sigma_dist}},
        },
    }
