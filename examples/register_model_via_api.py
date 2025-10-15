"""Example client for working with the Model Zoo registry API.

The script demonstrates how to:

1. Register a configuration JSON with `/registry/configs`.
2. Fetch the persisted model spec.
3. (Optional) Submit a synthetic run request via `/runs` using the registered
   configuration and display the evaluation summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib import error, request, parse


def _generate_linear(points: int, slope: float, intercept: float, noise_sigma: float) -> Tuple[list[float], list[float]]:
    import random

    random.seed(12345)
    x_vals = [i / 10 for i in range(1, points + 1)]
    y_vals = [intercept + slope * x + random.gauss(0, noise_sigma) for x in x_vals]
    return x_vals, y_vals


def _post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def _get_json(url: str) -> Dict[str, Any]:
    with request.urlopen(url) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def _poll_run(base_url: str, run_id: str, timeout: float = 600.0, interval: float = 0.5) -> Dict[str, Any]:
    status_url = parse.urljoin(base_url.rstrip("/") + "/", f"runs/{run_id}")
    deadline = time.time() + timeout
    while True:
        status_payload = _get_json(status_url)
        status = status_payload.get("status")
        if status in {"succeeded", "failed", "cancelled"}:
            return status_payload
        if time.time() >= deadline:
            raise TimeoutError(f"Run {run_id} did not finish within {timeout} seconds")
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a model configuration via the Model Zoo API")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL for the API (default: http://localhost:8000)")
    parser.add_argument("--config", default="configs/linear_regression.json", help="Path to the configuration JSON file")
    parser.add_argument("--env", default=None, help="Environment label to associate with the configuration")
    parser.add_argument("--model-id", default=None, help="Override model id if different from config")
    parser.add_argument("--skip-fetch", action="store_true", help="Do not fetch the model spec after registration")
    parser.add_argument("--run", action="store_true", help="Submit a run request using synthetic linear data")
    parser.add_argument("--points", type=int, default=100, help="Number of synthetic points when --run is used")
    parser.add_argument("--slope", type=float, default=2.0, help="Slope for synthetic data when --run is used")
    parser.add_argument("--intercept", type=float, default=1.0, help="Intercept for synthetic data when --run is used")
    parser.add_argument("--noise", type=float, default=0.1, help="Noise sigma for synthetic data when --run is used")
    parser.add_argument("--coeff-table", type=Path, default=None, help="Embed coefficient priors from a CSV before registering")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Configuration file not found: {config_path}")

    config_payload = json.loads(config_path.read_text())

    if args.coeff_table:
        if not args.coeff_table.exists():
            raise SystemExit(f"Coefficient table not found: {args.coeff_table}")
        with args.coeff_table.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            table_rows: List[Dict[str, Any]] = [
                {k: (v if v != "" else None) for k, v in row.items()} for row in reader
            ]
        priors_section = config_payload.setdefault("coefficient_priors", {})
        priors_section["tables"] = [{"content": table_rows}]

    payload = {
        "env": args.env,
        "model_id": args.model_id,
        "config": config_payload,
    }

    endpoint = parse.urljoin(args.base_url.rstrip("/") + "/", "registry/configs")
    try:
        response = _post_json(endpoint, payload)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Failed to register configuration ({exc.code}): {detail}")

    model_id = response.get("model_id") or args.model_id
    print("Registered model:")
    print(json.dumps(response, indent=2))

    if not args.skip_fetch and model_id:
        spec_url = parse.urljoin(args.base_url.rstrip("/") + "/", f"registry/models/{model_id}")
        try:
            spec = _get_json(spec_url)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SystemExit(f"Failed to fetch model spec ({exc.code}): {detail}")
        print("Fetched stored spec:")
        print(json.dumps(spec, indent=2))

    if args.run and model_id:
        x_vals, y_vals = _generate_linear(args.points, args.slope, args.intercept, args.noise)
        run_payload = {
            "model_id": model_id,
            "env": args.env,
            "data_sources": {
                "x_values": x_vals,
                "y_values": y_vals,
            },
            "evaluation": {
                "metric_ids": ["r2_v1", "mse_v1"],
                "y_true_key": "y"
            }
        }
        runs_endpoint = parse.urljoin(args.base_url.rstrip("/") + "/", "runs")
        try:
            submission = _post_json(runs_endpoint, run_payload)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SystemExit(f"Run submission failed ({exc.code}): {detail}")

        run_ticket = submission.get("run_id")
        print(f"Submitted run: {run_ticket} (status={submission.get('status')})")
        status_payload = _poll_run(args.base_url, run_ticket)
        print("Run status:")
        print(json.dumps(status_payload, indent=2))

        result_payload = status_payload.get("result") or {}
        if result_payload:
            print("Diagnostics (coefficients):")
            print(json.dumps(result_payload.get("diagnostics", {}).get("coefficients_model", {}), indent=2))


if __name__ == "__main__":
    main()
