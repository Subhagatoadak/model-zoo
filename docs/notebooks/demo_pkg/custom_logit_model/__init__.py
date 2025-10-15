from __future__ import annotations
from typing import Any, Dict
import numpy as np
from sklearn.linear_model import LogisticRegression

def train_and_classify(inputs: Dict[str, Any], params: Dict[str, Any], context: Any) -> Dict[str, Any]:
    X = np.asarray(inputs['X'], dtype=float)
    y = np.asarray(inputs['y'], dtype=int)
    solver = params.get('solver', 'lbfgs')
    max_iter = int(params.get('max_iter', 200))
    clf = LogisticRegression(solver=solver, max_iter=max_iter)
    clf.fit(X, y)
    predictions = clf.predict(X).tolist()
    probabilities = clf.predict_proba(X).tolist()
    diagnostics = {
        'coef': clf.coef_.tolist(),
        'intercept': clf.intercept_.tolist(),
        'classes': clf.classes_.tolist(),
    }
    return {
        'predictions': predictions,
        'artifacts': {'probabilities': probabilities},
        'diagnostics': diagnostics,
    }
