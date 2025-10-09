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

    if fit_intercept:
        # OLS with intercept via covariance/variance
        beta = _cov(X, y) / (_var(X) + 1e-12)
        alpha = _mean(y) - beta * _mean(X)
    else:
        # Force intercept to 0
        # beta = (X^T y) / (X^T X)
        num = sum(X[i] * y[i] for i in range(len(X)))
        den = sum(x * x for x in X) + 1e-12
        beta, alpha = num / den, 0.0

    y_pred = [alpha + beta * x for x in X]

    # R^2
    y_mean = _mean(y)
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum((yi - yhat) ** 2 for yi, yhat in zip(y, y_pred))
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12)) if ss_tot > 0 else 0.0

    return {
        "predictions": y_pred,
        "diagnostics": {"r2": r2, "slope": beta, "intercept": alpha},
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
    prior_tau = float(params.get("prior_tau", 10.0))
    if sigma <= 0 or prior_tau <= 0:
        raise ValueError("'sigma' and 'prior_tau' must be positive")

    n = len(X)
    mx, my = _mean(X), _mean(y)
    # Center if intercept included
    Xc = [x - mx for x in X] if fit_intercept else X
    yc = [yi - my for yi in y] if fit_intercept else y

    sxx = sum(x * x for x in Xc)
    sxy = sum(Xc[i] * yc[i] for i in range(n))

    # Posterior for beta: Normal(mn, vn)
    v0 = prior_tau ** 2
    vn = 1.0 / (1.0 / v0 + sxx / (sigma ** 2 + 1e-12))
    mn = vn * (sxy / (sigma ** 2 + 1e-12))

    beta_map = mn  # posterior mean == MAP for Gaussian-Gaussian
    alpha_map = (my - beta_map * mx) if fit_intercept else 0.0

    y_pred = [alpha_map + beta_map * x for x in X]

    return {
        "predictions": y_pred,
        "diagnostics": {
            "posterior_mean_beta": beta_map,
            "posterior_var_beta": vn,
            "alpha": alpha_map,
        },
        "metadata": {
            "model": "bayesian_linear_regression",
            "sigma": sigma,
            "prior_tau": prior_tau,
            "fit_intercept": fit_intercept,
        },
    }


def make_synthetic_linear(n: int = 100, slope: float = 2.0, intercept: float = 1.0, noise_sigma: float = 0.5) -> Tuple[List[float], List[float]]:
    import random
    random.seed(1234)
    X = [(i + 1) / 10.0 for i in range(n)]
    y = [intercept + slope * x + random.gauss(0, noise_sigma) for x in X]
    return X, y

