from __future__ import annotations
from typing import Dict, Any

def r_style_summary_string(summary: Dict[str, Any]) -> str:
    """Render a compact, R-style summary string from ModelRunResult.summary.

    Expected keys in `summary`:
    - Call: dict with 'module', 'callable', and optionally 'params'
    - Coefficients: dict of name -> estimate
    - Fit: dict with 'r_squared' and 'mse' when available
    - Sizes: dict with 'n', 'n_train', 'n_valid'
    - Metrics: {'train': {...}, 'validation': {...}}
    """
    if not summary:
        return "<no summary available>"

    lines = []

    # Call
    call = summary.get("Call", {}) or {}
    if isinstance(call, dict):
        module = call.get("module")
        func = call.get("callable")
        params = call.get("params") or {}
        callee = None
        if module and func:
            callee = f"{module}::{func}"
        elif func:
            callee = func
        elif module:
            callee = module
        if callee:
            lines.append(f"Call:\n  {callee}(")
            if params:
                for k, v in params.items():
                    lines.append(f"    {k} = {repr(v)}")
            lines.append(")")
        else:
            lines.append("Call: <unknown>")
    else:
        lines.append("Call: <unknown>")

    # Coefficients table (Estimate only)
    coeffs = summary.get("Coefficients", {}) or {}
    if isinstance(coeffs, dict) and coeffs:
        lines.append("")
        lines.append("Coefficients:")
        # Determine column widths
        name_w = max([len("(Intercept)")] + [len(str(k)) for k in coeffs.keys()])
        est_w = max(len("Estimate"), max(len(f"{float(v):.6f}") for v in coeffs.values())) if coeffs else len("Estimate")
        header = f"  {'':<{name_w}}  {'Estimate':>{est_w}}"
        lines.append(header)
        lines.append(f"  {'-'*name_w}  {'-'*est_w}")
        for k, v in coeffs.items():
            try:
                val = float(v)
            except Exception:
                val = v
            lines.append(f"  {str(k):<{name_w}}  {val:>{est_w}.6f}" if isinstance(val, float) else f"  {str(k):<{name_w}}  {str(val):>{est_w}}")

    # Fit stats
    fit = summary.get("Fit", {}) or {}
    if fit:
        lines.append("")
        parts = []
        if "r_squared" in fit and fit["r_squared"] is not None:
            parts.append(f"R-squared: {fit['r_squared']:.6f}")
        if "mse" in fit and fit["mse"] is not None:
            parts.append(f"MSE: {fit['mse']:.6f}")
        if parts:
            lines.append("  " + ", ".join(parts))

    # Sizes
    sizes = summary.get("Sizes", {}) or {}
    if sizes:
        sz_parts = []
        for key in ("n", "n_train", "n_valid"):
            if sizes.get(key) is not None:
                sz_parts.append(f"{key}={sizes[key]}")
        if sz_parts:
            lines.append("  " + ", ".join(sz_parts))

    # Metrics
    metrics = summary.get("Metrics", {}) or {}
    if metrics:
        train = metrics.get("train") or {}
        valid = metrics.get("validation") or {}
        if train:
            lines.append("")
            lines.append("Train metrics:")
            for k, v in train.items():
                try:
                    lines.append(f"  {k}: {float(v):.6f}")
                except Exception:
                    lines.append(f"  {k}: {v}")
        if valid:
            lines.append("")
            lines.append("Validation metrics:")
            for k, v in valid.items():
                try:
                    lines.append(f"  {k}: {float(v):.6f}")
                except Exception:
                    lines.append(f"  {k}: {v}")

    return "\n".join(lines)

