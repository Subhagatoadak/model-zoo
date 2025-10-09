from __future__ import annotations
from typing import Any, Dict, List


def sk_lr_train_and_infer(inputs: Dict[str, Any], params: Dict[str, Any], context) -> Dict[str, Any]:
    """Train and infer using scikit-learn LinearRegression (single feature or multi-feature).

    Requires scikit-learn and numpy to be installed.
    inputs: 'X' as list[float] or list[list[float]], 'y' as list[float]
    params: {'fit_intercept': bool, 'train_indices': Optional[list[int]]}
    """
    try:
        import numpy as np
        from sklearn.linear_model import LinearRegression
    except Exception as e:
        raise RuntimeError("scikit-learn and numpy are required for sk_lr_train_and_infer") from e

    X = inputs.get("X")
    y = inputs.get("y")
    if X is None or y is None:
        raise ValueError("inputs must contain 'X' and 'y'")

    # Normalize shapes
    if isinstance(X[0], (int, float)):
        X_arr = np.array(X, dtype=float).reshape(-1, 1)
    else:
        X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=float)

    # Train subset if provided
    train_idx = params.get("train_indices")
    if train_idx is not None:
        X_train = X_arr[train_idx]
        y_train = y_arr[train_idx]
    else:
        X_train, y_train = X_arr, y_arr

    fit_intercept = bool(params.get("fit_intercept", True))
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_arr)
    diagnostics: Dict[str, Any] = {
        "coef": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "fit_intercept": fit_intercept,
    }

    return {
        "predictions": y_pred.tolist(),
        "diagnostics": diagnostics,
        "metadata": {"model": "sklearn_linear_regression"},
    }

