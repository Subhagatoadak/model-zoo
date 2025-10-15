"""PyMC model callables for in‑process execution via the Library runner.

Contains:
- `pymc_linear_nuts`: linear regression with per‑feature priors and NUTS sampling
- `pymc_hierarchical_mmm`: hierarchical MMM with geo/brand random intercepts
"""
from __future__ import annotations
from typing import Any, Dict, List


def pymc_linear_nuts(inputs: Dict[str, Any], params: Dict[str, Any], context) -> Dict[str, Any]:
    """Bayesian linear regression via PyMC NUTS sampler.

    Hyperparameters (draws, tune, etc.) live under ``parameters``; coefficient
    priors are supplied separately in ``coefficients`` with intercept + variables.
    Legacy ``priors`` dictionaries are still supported for backwards
    compatibility.
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

    coeff_cfg = params.get("coefficients") if isinstance(params.get("coefficients"), dict) else {}
    priors = params.get("priors", {}) or {}
    use_coefficients = bool(coeff_cfg)

    sigma_dist = str(params.get("sigma_distribution", "half_normal")).lower()
    sigma_scale = float(params.get("sigma_scale", 1.0))
    sigma_shape = float(params.get("sigma_shape", 2.0))
    sigma_rate = float(params.get("sigma_rate", 1.0))

    if not use_coefficients:
        sigma_prior = priors.get("sigma", {})
        if sigma_prior:
            sigma_dist = str(sigma_prior.get("distribution", sigma_dist)).lower()
            sigma_scale = float(sigma_prior.get("scale", sigma_prior.get("sigma", sigma_scale)))
            sigma_shape = float(sigma_prior.get("shape", sigma_prior.get("alpha", sigma_shape)))
            sigma_rate = float(sigma_prior.get("rate", sigma_prior.get("beta", sigma_rate)))

    p = X_arr.shape[1]

    with pm.Model() as model:
        # Intercept prior
        if use_coefficients and coeff_cfg.get("intercept"):
            intercept_cfg = coeff_cfg["intercept"] or {}
            intercept_dist = str(intercept_cfg.get("distribution", "normal")).lower()
            if intercept_dist in ("normal",):
                alpha_mu = float(intercept_cfg.get("mu", intercept_cfg.get("mean", 0.0)))
                alpha_sigma = float(intercept_cfg.get("sigma", intercept_cfg.get("tau", 10.0)))
                alpha = pm.Normal("alpha", mu=alpha_mu, sigma=alpha_sigma)
            elif intercept_dist in ("half_normal", "halfnormal"):
                alpha_sigma = float(intercept_cfg.get("sigma", intercept_cfg.get("scale", 10.0)))
                alpha = pm.HalfNormal("alpha", sigma=alpha_sigma)
            else:
                raise ValueError(f"Unsupported intercept distribution: {intercept_dist}")
        else:
            alpha = pm.Normal("alpha", mu=0.0, sigma=10.0)

        beta_vars: List[Any] = []

        if use_coefficients:
            variable_cfg: Dict[str, Dict[str, Any]] = {}
            for entry in coeff_cfg.get("variables", []) or []:
                if isinstance(entry, dict) and entry.get("name"):
                    variable_cfg[str(entry["name"])] = entry

            for name in feature_names:
                spec = variable_cfg.get(name, {})
                dist_i = str(spec.get("distribution", "normal")).lower()
                rv_name = f"beta_{name}".replace(" ", "_").replace("[", "_").replace("]", "_")
                if dist_i in ("normal",):
                    mu_i = float(spec.get("mu", spec.get("mean", 0.0)))
                    sigma_i = float(spec.get("sigma", spec.get("tau", 10.0)))
                    beta_vars.append(pm.Normal(rv_name, mu=mu_i, sigma=sigma_i))
                elif dist_i in ("half_normal", "halfnormal"):
                    sigma_i = float(spec.get("sigma", spec.get("scale", 1.0)))
                    beta_vars.append(pm.HalfNormal(rv_name, sigma=sigma_i))
                elif dist_i in ("gamma", "erlang"):
                    shape_i = float(spec.get("shape", 2.0))
                    rate_i = float(spec.get("rate", 1.0))
                    beta_vars.append(pm.Gamma(rv_name, alpha=shape_i, beta=rate_i))
                else:
                    raise ValueError(f"Unsupported coefficient distribution for '{name}': {dist_i}")
        else:
            beta_prior = priors.get("beta", {})
            beta_dist_raw = beta_prior.get("distribution", "normal")

            def _legacy_param(name: str, default: float, i: int) -> float:
                val = beta_prior.get(name, None)
                if isinstance(val, list):
                    if len(val) != p:
                        raise ValueError(f"beta.{name} length must equal number of features")
                    return float(val[i]) if val[i] is not None else float(default)
                if val is None:
                    return float(default)
                return float(val)

            for i, name in enumerate(feature_names):
                if isinstance(beta_dist_raw, list):
                    if len(beta_dist_raw) != p:
                        raise ValueError("beta.distribution length must equal number of features")
                    dist_i = str(beta_dist_raw[i]).lower()
                else:
                    dist_i = str(beta_dist_raw).lower()
                rv_name = f"beta_{name}".replace(" ", "_").replace("[", "_").replace("]", "_")
                if dist_i in ("normal",):
                    mu_i = _legacy_param("mu", 0.0, i)
                    tau_i = _legacy_param("tau", 10.0, i)
                    beta_vars.append(pm.Normal(rv_name, mu=mu_i, sigma=tau_i))
                elif dist_i in ("half_normal", "halfnormal"):
                    scale_i = _legacy_param("scale", 1.0, i)
                    beta_vars.append(pm.HalfNormal(rv_name, sigma=scale_i))
                elif dist_i in ("gamma", "erlang"):
                    shape_i = _legacy_param("shape", 2.0, i)
                    rate_i = _legacy_param("rate", 1.0, i)
                    beta_vars.append(pm.Gamma(rv_name, alpha=shape_i, beta=rate_i))
                else:
                    raise ValueError(f"Unsupported beta prior distribution: {dist_i}")

        beta_vec = pm.math.stack(beta_vars)

        if sigma_dist in ("half_normal", "halfnormal"):
            sigma = pm.HalfNormal("sigma", sigma=sigma_scale)
        elif sigma_dist in ("gamma", "erlang"):
            sigma = pm.Gamma("sigma", alpha=sigma_shape, beta=max(1e-6, sigma_rate))
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

    import numpy as _np
    alpha_mean = float(idata.posterior["alpha"].values.mean())
    beta_means: List[float] = []
    for name in feature_names:
        rv_name = f"beta_{name}".replace(" ", "_").replace("[", "_").replace("]", "_")
        beta_means.append(float(idata.posterior[rv_name].values.mean()))

    beta_mean_vec = _np.array(beta_means)
    y_pred = (alpha_mean + X_arr.dot(beta_mean_vec)).tolist()

    diagnostics: Dict[str, Any] = {
        "coefficients_model": {
            "alpha": alpha_mean,
            "dv_sigma": float(idata.posterior["sigma"].values.mean())
        },
        "param_values": {
            "draws": draws,
            "tune": tune,
            "target_accept": target_accept,
            "chains": chains,
        },
    }
    for i, name in enumerate(feature_names):
        diagnostics["coefficients_model"][f"coef[{name}]"] = float(beta_mean_vec[i])

    metadata_coeffs = coeff_cfg if use_coefficients else {"legacy_priors": priors.get("beta")}
    sigma_meta = {"distribution": sigma_dist}
    if sigma_dist in ("half_normal", "halfnormal"):
        sigma_meta["scale"] = sigma_scale
    else:
        sigma_meta["shape"] = sigma_shape
        sigma_meta["rate"] = sigma_rate

    return {
        "predictions": y_pred,
        "diagnostics": diagnostics,
        "metadata": {
            "model": "pymc_linear_nuts",
            "feature_names": feature_names,
            "coefficients": metadata_coeffs,
            "sigma": sigma_meta,
        },
    }


def pymc_hierarchical_mmm(inputs: Dict[str, Any], params: Dict[str, Any], context) -> Dict[str, Any]:
    """Hierarchical (partial-pooling) Bayesian marketing mix model via PyMC NUTS.

    Expects X as list[dict] with spend features and categorical keys for geography and brand.
    - Features (continuous): defined by `feature_names` param (e.g., ["tv", "radio", "social"]).
    - Categorical: `geo_key` (default: "geo"), `brand_key` (default: "brand").

    Priors:
      - Per-feature beta can be Normal/HalfNormal/Gamma per-feature (same schema as pymc_linear_nuts)
      - Global intercept alpha ~ Normal(0,10)
      - Group intercepts: alpha_geo[g] ~ Normal(0, sigma_geo); sigma_geo ~ HalfNormal(2)
                          alpha_brand[b] ~ Normal(0, sigma_brand); sigma_brand ~ HalfNormal(2)
      - Observation sigma: half_normal | gamma | erlang
    """
    try:
        import numpy as np
        import pymc as pm
    except Exception as e:
        raise RuntimeError("pymc and numpy are required for pymc_hierarchical_mmm") from e

    X_rows = inputs.get("X")
    y = inputs.get("y")
    if X_rows is None or y is None:
        raise ValueError("inputs must contain 'X' and 'y'")
    if not isinstance(X_rows, list) or not isinstance(X_rows[0], dict):
        raise ValueError("X must be list[dict] with spend features and categorical fields")

    feature_names: List[str] = params.get("feature_names") or [k for k in X_rows[0].keys() if k not in ("geo", "brand")]
    geo_key = params.get("geo_key", "geo")
    brand_key = params.get("brand_key", "brand")

    # Build matrices and categorical index vectors
    n = len(X_rows)
    X_mat = np.array([[float(row[f]) for f in feature_names] for row in X_rows], dtype=float)
    y_arr = np.array(y, dtype=float)

    # Categorical indices
    geos = sorted({row[geo_key] for row in X_rows})
    brands = sorted({row[brand_key] for row in X_rows})
    geo_index = {g: i for i, g in enumerate(geos)}
    brand_index = {b: i for i, b in enumerate(brands)}
    geo_idx = np.array([geo_index[row[geo_key]] for row in X_rows], dtype=int)
    brand_idx = np.array([brand_index[row[brand_key]] for row in X_rows], dtype=int)

    # Optional: train subset
    train_idx = params.get("train_indices")
    if train_idx is not None:
        X_train = X_mat[train_idx]
        y_train = y_arr[train_idx]
        geo_train = geo_idx[train_idx]
        brand_train = brand_idx[train_idx]
    else:
        X_train, y_train, geo_train, brand_train = X_mat, y_arr, geo_idx, brand_idx

    draws = int(params.get("draws", 1000))
    tune = int(params.get("tune", 1000))
    target_accept = float(params.get("target_accept", 0.9))
    chains = int(params.get("chains", 1))
    cores = int(params.get("cores", 1))
    progressbar = bool(params.get("progressbar", False))
    compute_convergence_checks = bool(params.get("compute_convergence_checks", False))

    coeff_cfg = params.get("coefficients") if isinstance(params.get("coefficients"), dict) else {}
    priors = params.get("priors", {}) or {}
    use_coefficients = bool(coeff_cfg)

    sigma_dist = str(params.get("sigma_distribution", "half_normal")).lower()
    sigma_scale = float(params.get("sigma_scale", 5.0))
    sigma_shape = float(params.get("sigma_shape", 2.0))
    sigma_rate = float(params.get("sigma_rate", 1.0))

    if not use_coefficients:
        sigma_prior = priors.get("sigma", {})
        if sigma_prior:
            sigma_dist = str(sigma_prior.get("distribution", sigma_dist)).lower()
            sigma_scale = float(sigma_prior.get("scale", sigma_prior.get("sigma", sigma_scale)))
            sigma_shape = float(sigma_prior.get("shape", sigma_prior.get("alpha", sigma_shape)))
            sigma_rate = float(sigma_prior.get("rate", sigma_prior.get("beta", sigma_rate)))

    p = X_mat.shape[1]

    with pm.Model() as model:
        if use_coefficients and coeff_cfg.get("intercept"):
            intercept_cfg = coeff_cfg["intercept"] or {}
            intercept_dist = str(intercept_cfg.get("distribution", "normal")).lower()
            if intercept_dist in ("normal",):
                alpha_mu = float(intercept_cfg.get("mu", intercept_cfg.get("mean", 0.0)))
                alpha_sigma = float(intercept_cfg.get("sigma", intercept_cfg.get("tau", 10.0)))
                alpha = pm.Normal("alpha", mu=alpha_mu, sigma=alpha_sigma)
            elif intercept_dist in ("half_normal", "halfnormal"):
                alpha_sigma = float(intercept_cfg.get("sigma", intercept_cfg.get("scale", 10.0)))
                alpha = pm.HalfNormal("alpha", sigma=alpha_sigma)
            else:
                raise ValueError(f"Unsupported intercept distribution: {intercept_dist}")
        else:
            alpha = pm.Normal("alpha", mu=0.0, sigma=10.0)
        # Hierarchical intercepts
        sigma_geo = pm.HalfNormal("sigma_geo", sigma=2.0)
        sigma_brand = pm.HalfNormal("sigma_brand", sigma=2.0)
        alpha_geo = pm.Normal("alpha_geo", mu=0.0, sigma=sigma_geo, shape=len(geos))
        alpha_brand = pm.Normal("alpha_brand", mu=0.0, sigma=sigma_brand, shape=len(brands))

        # Per-feature betas
        beta_vars = []
        if use_coefficients:
            variable_cfg: Dict[str, Dict[str, Any]] = {}
            for entry in coeff_cfg.get("variables", []) or []:
                if isinstance(entry, dict) and entry.get("name"):
                    variable_cfg[str(entry["name"])] = entry
            for name in feature_names:
                spec = variable_cfg.get(name, {})
                dist_i = str(spec.get("distribution", "normal")).lower()
                rv_name = f"beta_{name}".replace(" ", "_").replace("[", "_").replace("]", "_")
                if dist_i in ("normal",):
                    mu_i = float(spec.get("mu", spec.get("mean", 0.0)))
                    sigma_i = float(spec.get("sigma", spec.get("tau", 10.0)))
                    beta_vars.append(pm.Normal(rv_name, mu=mu_i, sigma=sigma_i))
                elif dist_i in ("half_normal", "halfnormal"):
                    sigma_i = float(spec.get("sigma", spec.get("scale", 1.0)))
                    beta_vars.append(pm.HalfNormal(rv_name, sigma=sigma_i))
                elif dist_i in ("gamma", "erlang"):
                    shape_i = float(spec.get("shape", 2.0))
                    rate_i = float(spec.get("rate", 1.0))
                    beta_vars.append(pm.Gamma(rv_name, alpha=shape_i, beta=rate_i))
                else:
                    raise ValueError(f"Unsupported coefficient distribution for '{name}': {dist_i}")
        else:
            beta_prior = priors.get("beta", {})
            beta_dist_raw = beta_prior.get("distribution", "normal")

            def _legacy_param(name: str, default: float, i: int) -> float:
                val = beta_prior.get(name, None)
                if isinstance(val, list):
                    if len(val) != p:
                        raise ValueError(f"beta.{name} length must equal number of features")
                    return float(val[i]) if val[i] is not None else float(default)
                if val is None:
                    return float(default)
                return float(val)

            for i, name in enumerate(feature_names):
                if isinstance(beta_dist_raw, list):
                    if len(beta_dist_raw) != p:
                        raise ValueError("beta.distribution length must equal number of features")
                    dist_i = str(beta_dist_raw[i]).lower()
                else:
                    dist_i = str(beta_dist_raw).lower()
                rv_name = f"beta_{name}".replace(" ", "_").replace("[", "_").replace("]", "_")
                if dist_i in ("normal",):
                    mu_i = _legacy_param("mu", 0.0, i)
                    tau_i = _legacy_param("tau", 10.0, i)
                    beta_vars.append(pm.Normal(rv_name, mu=mu_i, sigma=tau_i))
                elif dist_i in ("half_normal", "halfnormal"):
                    scale_i = _legacy_param("scale", 1.0, i)
                    beta_vars.append(pm.HalfNormal(rv_name, sigma=scale_i))
                elif dist_i in ("gamma", "erlang"):
                    shape_i = _legacy_param("shape", 2.0, i)
                    rate_i = _legacy_param("rate", 1.0, i)
                    beta_vars.append(pm.Gamma(rv_name, alpha=shape_i, beta=rate_i))
                else:
                    raise ValueError(f"Unsupported beta prior distribution: {dist_i}")
        beta_vec = pm.math.stack(beta_vars)

        # Observation sigma prior
        if sigma_dist in ("half_normal", "halfnormal"):
            sigma = pm.HalfNormal("sigma", sigma=sigma_scale)
        elif sigma_dist in ("gamma", "erlang"):
            sigma = pm.Gamma("sigma", alpha=sigma_shape, beta=max(1e-6, sigma_rate))
        else:
            raise ValueError(f"Unsupported sigma prior distribution: {sigma_dist}")

        mu = alpha + alpha_geo[geo_train] + alpha_brand[brand_train] + pm.math.dot(X_train, beta_vec)
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

    # Posterior means for predictions and diagnostics
    import numpy as _np
    alpha_mean = float(idata.posterior["alpha"].values.mean())
    sigma_geo_mean = float(idata.posterior["sigma_geo"].values.mean())
    sigma_brand_mean = float(idata.posterior["sigma_brand"].values.mean())
    alpha_geo_mean = _np.array(idata.posterior["alpha_geo"].values).mean(axis=(0, 1))
    alpha_brand_mean = _np.array(idata.posterior["alpha_brand"].values).mean(axis=(0, 1))

    beta_means: List[float] = []
    for name in feature_names:
        rv_name = f"beta_{name}".replace(" ", "_").replace("[", "_").replace("]", "_")
        beta_means.append(float(idata.posterior[rv_name].values.mean()))
    beta_mean_vec = _np.array(beta_means)

    # Predictions for all rows
    y_pred = (
        alpha_mean
        + alpha_geo_mean[geo_idx]
        + alpha_brand_mean[brand_idx]
        + X_mat.dot(beta_mean_vec)
    ).tolist()

    diagnostics: Dict[str, Any] = {
        "coefficients_model": {
            "alpha": alpha_mean,
            "sigma_geo": sigma_geo_mean,
            "sigma_brand": sigma_brand_mean,
            "dv_sigma": float(idata.posterior["sigma"].values.mean()),
        },
        "param_values": {
            "draws": draws,
            "tune": tune,
            "target_accept": target_accept,
            "chains": chains,
        },
    }
    for i, name in enumerate(feature_names):
        diagnostics["coefficients_model"][f"coef[{name}]"] = float(beta_mean_vec[i])
    # Include group intercept summaries for interpretability
    for i, g in enumerate(geos):
        diagnostics["coefficients_model"][f"alpha_geo[{g}]"] = float(alpha_geo_mean[i])
    for i, b in enumerate(brands):
        diagnostics["coefficients_model"][f"alpha_brand[{b}]"] = float(alpha_brand_mean[i])

    sigma_meta = {"distribution": sigma_dist}
    if sigma_dist in ("half_normal", "halfnormal"):
        sigma_meta["scale"] = sigma_scale
    else:
        sigma_meta["shape"] = sigma_shape
        sigma_meta["rate"] = sigma_rate

    return {
        "predictions": y_pred,
        "diagnostics": diagnostics,
        "metadata": {
            "model": "pymc_hierarchical_mmm",
            "feature_names": feature_names,
            "geo_key": geo_key,
            "brand_key": brand_key,
            "geos": geos,
            "brands": brands,
            "coefficients": coeff_cfg if use_coefficients else {"legacy_priors": priors.get("beta")},
            "sigma": sigma_meta,
        },
    }
