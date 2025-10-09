from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math

# Simple utilities kept dependency-free (no NumPy required)

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def _var(xs: List[float]) -> float:
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs) if xs else 0.0

def _cov(xs: List[float], ys: List[float]) -> float:
    mx, my = _mean(xs), _mean(ys)
    n = len(xs)
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / n if n else 0.0


def linreg_train_and_infer(inputs: Dict[str, Any], params: Dict[str, Any], context) -> Dict[str, Any]:
    """Deterministic single-feature linear regression using closed-form OLS.

    inputs: expects keys 'X' (list[float]) and 'y' (list[float])
    params:
      - fit_intercept: bool = True
    returns dict with predictions, diagnostics, metadata
    """
    X: List[float] = inputs.get("X", [])
    y: List[float] = inputs.get("y", [])
    fit_intercept = bool(params.get("fit_intercept", True))

    if not X or not y or len(X) != len(y):
        raise ValueError("linreg: inputs must provide 'X' and 'y' of same non-zero length")

    # Optional: train on subset of indices if provided
    train_idx = params.get("train_indices")
    if train_idx is not None:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
    else:
        X_train, y_train = X, y

    if fit_intercept:
        # OLS with intercept via covariance/variance
        beta = _cov(X_train, y_train) / (_var(X_train) + 1e-12)
        alpha = _mean(y_train) - beta * _mean(X_train)
    else:
        # Force intercept to 0
        # beta = (X^T y) / (X^T X)
        num = sum(X_train[i] * y_train[i] for i in range(len(X_train)))
        den = sum(x * x for x in X_train) + 1e-12
        beta, alpha = num / den, 0.0

    y_pred = [alpha + beta * x for x in X]

    return {
        "predictions": y_pred,
        # Keep coefficients here; evaluation metrics are computed separately
        "diagnostics": {"slope": beta, "intercept": alpha},
        "metadata": {"model": "linear_regression", "fit_intercept": fit_intercept},
    }


def bayes_linreg_train_and_infer(inputs: Dict[str, Any], params: Dict[str, Any], context) -> Dict[str, Any]:
    """Conjugate Bayesian linear regression (single feature) with Normal prior.

    Prior: beta ~ Normal(0, tau0^2)
    Likelihood: y_i = alpha + beta x_i + eps, eps ~ Normal(0, sigma^2)
    We center X and y for stable alpha estimate: alpha = mean(y) - beta * mean(X)

    params:
      - sigma: observation noise std (float, required)
      - prior_tau: prior std for beta (float, default 10.0)
      - fit_intercept: bool (default True)
    returns dict with predictions and posterior summaries
    """
    X: List[float] = inputs.get("X", [])
    y: List[float] = inputs.get("y", [])
    if not X or not y or len(X) != len(y):
        raise ValueError("bayes_linreg: inputs must provide 'X' and 'y' of same non-zero length")

    fit_intercept = bool(params.get("fit_intercept", True))
    sigma = float(params.get("sigma", 1.0))
    prior_tau = float(params.get("prior_tau", params.get("prior_std", 10.0)))
    prior_mu = float(params.get("prior_mu", 0.0))
    draws = int(params.get("draws", 0))  # number of posterior draws to return (optional)
    _tune = int(params.get("tune", 0))   # accepted but unused in conjugate demo
    # Priors format (example):
    # priors = {
    #   "beta": {"distribution": "normal", "mu": 0.0, "tau": 5.0},
    #   "alpha": {"distribution": "normal", "mu": 0.0, "tau": 10.0},  # (alpha prior currently informational)
    # }
    priors = params.get("priors", {})     # optional dict of distributions
    if isinstance(priors, dict):
        b = priors.get("beta", {}) or {}
        dist = (b.get("distribution") or "normal").lower()
        if dist != "normal":
            raise ValueError(f"Unsupported prior distribution for beta: {dist}")
        if "mu" in b: prior_mu = float(b["mu"])
        if "tau" in b: prior_tau = float(b["tau"])
    if sigma <= 0 or prior_tau <= 0:
        raise ValueError("'sigma' and 'prior_tau' must be positive")

    # Optional: train on subset of indices
    train_idx = params.get("train_indices")
    if train_idx is not None:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
    else:
        X_train, y_train = X, y

    n = len(X_train)
    mx, my = _mean(X_train), _mean(y_train)
    # Center if intercept included
    Xc = [x - mx for x in X_train] if fit_intercept else X_train
    yc = [yi - my for yi in y_train] if fit_intercept else y_train

    sxx = sum(x * x for x in Xc)
    sxy = sum(Xc[i] * yc[i] for i in range(n))

    # Conjugate updates
    # Two cases: (A) known sigma; (B) Gamma prior on precision tau = 1/sigma^2
    pri_sigma = priors.get("sigma") if isinstance(priors, dict) else None
    posterior_info: Dict[str, Any] = {}
    if isinstance(pri_sigma, dict) and (pri_sigma.get("distribution") or "").lower() == "gamma":
        # Normal-Gamma prior: beta | tau ~ N(mu0, (tau*lambda0)^-1), tau ~ Gamma(a0, b0) (shape-rate)
        # Map prior_tau (std for beta under fixed sigma) to lambda0 by lambda0 = 1 / prior_tau^2
        mu0 = prior_mu
        lambda0 = 1.0 / (prior_tau ** 2 + 1e-12)
        a0 = float(pri_sigma.get("shape") or pri_sigma.get("alpha") or 1.0)
        b0 = float(pri_sigma.get("rate") or pri_sigma.get("beta") or 1.0)

        lambda_n = lambda0 + sxx
        mu_n = (lambda0 * mu0 + sxy) / (lambda_n + 1e-12)
        syy = sum(v * v for v in yc)
        a_n = a0 + n / 2.0
        b_n = b0 + 0.5 * (syy + lambda0 * mu0 * mu0 - lambda_n * mu_n * mu_n)

        # Posterior means
        tau_mean = a_n / (b_n + 1e-12)
        sigma_mean = (1.0 / (tau_mean + 1e-12)) ** 0.5
        beta_map = mu_n  # mean of marginal Student-t as well
        alpha_map = (my - beta_map * mx) if fit_intercept else 0.0

        # Marginal variance of beta (approx): b_n / (lambda_n * (a_n - 1)) when a_n > 1
        post_var_beta = b_n / (lambda_n * max(a_n - 1.0, 1.0))

        posterior_info.update({
            "posterior_beta_mean": mu_n,
            "posterior_beta_var": post_var_beta,
            "posterior_tau_shape": a_n,
            "posterior_tau_rate": b_n,
            "posterior_sigma_mean": sigma_mean,
        })
    else:
        # Known sigma; posterior for beta is Normal(mn, vn)
        v0 = prior_tau ** 2
        vn = 1.0 / (1.0 / v0 + sxx / (sigma ** 2 + 1e-12))
        mn = vn * (prior_mu / v0 + sxy / (sigma ** 2 + 1e-12))
        beta_map = mn
        alpha_map = (my - beta_map * mx) if fit_intercept else 0.0
        posterior_info.update({
            "posterior_beta_mean": beta_map,
            "posterior_beta_var": vn,
            "sigma": sigma,
        })

    y_pred = [alpha_map + beta_map * x for x in X]

    artifacts: Dict[str, Any] = {}
    if draws and draws > 0:
        import random
        random.seed(12345)
        # Draw posterior samples
        samples = []
        if isinstance(pri_sigma, dict) and (pri_sigma.get("distribution") or "").lower() == "gamma":
            # Sample tau ~ Gamma(a_n, b_n), then beta | tau ~ N(mu_n, 1/(tau*lambda_n))
            a_n = posterior_info["posterior_tau_shape"]
            b_n = posterior_info["posterior_tau_rate"]
            mu_n = posterior_info["posterior_beta_mean"]
            # Recompute lambda_n
            lambda0 = 1.0 / (prior_tau ** 2 + 1e-12)
            lambda_n = lambda0 + sxx
            for _ in range(draws):
                # Gamma sampling (shape-rate) via Marsaglia-Tsang for k>1 simplified? Use Python's random.gammavariate(k, theta=1/b)
                tau_s = random.gammavariate(a_n, 1.0 / (b_n + 1e-12))
                var_beta = 1.0 / (tau_s * (lambda_n + 1e-12))
                beta_s = random.gauss(mu_n, math.sqrt(max(var_beta, 1e-18)))
                alpha_s = (my - beta_s * mx) if fit_intercept else 0.0
                samples.append({"beta": beta_s, "alpha": alpha_s, "sigma": (1.0 / (tau_s + 1e-12)) ** 0.5})
        else:
            # Known sigma case
            # Use mn, vn from above scope
            v0 = prior_tau ** 2
            vn = 1.0 / (1.0 / v0 + sxx / (sigma ** 2 + 1e-12))
            mn = vn * (prior_mu / v0 + sxy / (sigma ** 2 + 1e-12))
            for _ in range(draws):
                beta_s = random.gauss(mn, math.sqrt(max(vn, 1e-18)))
                alpha_s = (my - beta_s * mx) if fit_intercept else 0.0
                samples.append({"beta": beta_s, "alpha": alpha_s})
        artifacts["posterior_samples"] = samples

    return {
        "predictions": y_pred,
        "diagnostics": {
            "beta": beta_map,
            "alpha": alpha_map,
            **posterior_info,
        },
        "metadata": {
            "model": "bayesian_linear_regression",
            "sigma": sigma,
            "prior_mu": prior_mu,
            "prior_tau": prior_tau,
            "fit_intercept": fit_intercept,
            "draws": draws,
        },
        "artifacts": artifacts,
    }


def make_synthetic_linear(n: int = 100, slope: float = 2.0, intercept: float = 1.0, noise_sigma: float = 0.5) -> Tuple[List[float], List[float]]:
    import random
    random.seed(1234)
    X = [(i + 1) / 10.0 for i in range(n)]
    y = [intercept + slope * x + random.gauss(0, noise_sigma) for x in X]
    return X, y
