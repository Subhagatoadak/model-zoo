from __future__ import annotations
from typing import Any, Dict, List
from ..utils.logging import get_logger


def sk_lr_train_and_infer(inputs: Dict[str, Any], params: Dict[str, Any], context) -> Dict[str, Any]:
    """Train and infer using scikit-learn LinearRegression (single feature or multi-feature).

    Requires scikit-learn and numpy to be installed.
    inputs: 'X' as list[float] or list[list[float]], 'y' as list[float]
    params: {'fit_intercept': bool, 'train_indices': Optional[list[int]]}
    """
    log = get_logger(__name__)
    try:
        import numpy as np
        from sklearn.linear_model import LinearRegression
    except Exception as e:
        raise RuntimeError("scikit-learn and numpy are required for sk_lr_train_and_infer") from e

    X = inputs.get("X")
    y = inputs.get("y")
    if X is None or y is None:
        raise ValueError("inputs must contain 'X' and 'y'")

    feature_names = params.get("feature_names")
    # Normalize shapes and derive feature names when possible
    if isinstance(X[0], dict):
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
        if feature_names is None:
            feature_names = [params.get("feature_name", "x")]
    else:
        X_arr = np.array(X, dtype=float)
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(X_arr.shape[1])]
    y_arr = np.array(y, dtype=float)

    # Train subset if provided
    train_idx = params.get("train_indices")
    if train_idx is not None:
        X_train = X_arr[train_idx]
        y_train = y_arr[train_idx]
    else:
        X_train, y_train = X_arr, y_arr

    # Heuristic: common misspelling guard (fit_interceopt / fit_uinterceopt)
    fit_intercept = params.get("fit_intercept")
    if fit_intercept is None:
        for typo in ("fit_interceopt", "fit_uinterceopt", "fitintercept"):
            if typo in params:
                log.warning("Parameter '%s' detected; did you mean 'fit_intercept'? Using its value.", typo)
                fit_intercept = params.get(typo)
                break
    if fit_intercept is None:
        fit_intercept = True
    fit_intercept = bool(fit_intercept)
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_arr)
    # Build named coefficients
    coefs = model.coef_.ravel().tolist()
    if len(feature_names) != len(coefs):
        if len(coefs) == 1 and len(feature_names) == 1:
            pass
        else:
            raise ValueError(
                f"Length of feature_names ({len(feature_names)}) does not match coefficients ({len(coefs)})"
            )
    diag: Dict[str, Any] = {f"coef[{name}]": float(coefs[i]) for i, name in enumerate(feature_names)}
    diag["intercept"] = float(model.intercept_)
    return {
        "predictions": y_pred.tolist(),
        # Only coefficient-like values in diagnostics; flags go to metadata
        "diagnostics": diag,
        "metadata": {
            "model": "sklearn_linear_regression",
            "fit_intercept": bool(fit_intercept),
            "feature_names": feature_names,
        },
    }
