"""Example client for registering model configurations via the FastAPI registry.

Usage
-----
Ensure the API is running (e.g. `docker compose up`). Then invoke:

    python examples/register_model_via_api.py \
        --config configs/linear_regression.json \
        --env dev --base-url http://localhost:8000

This script POSTs the configuration to `/registry/configs` and optionally
fetches the stored spec back from `/registry/models/{model_id}`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict
from urllib import error, request, parse


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a model configuration via the Model Zoo API")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL for the API (default: http://localhost:8000)")
    parser.add_argument("--config", default="configs/linear_regression.json", help="Path to the configuration JSON file")
    parser.add_argument("--env", default=None, help="Environment label to associate with the configuration")
    parser.add_argument("--model-id", default=None, help="Override model id if different from config")
    parser.add_argument("--skip-fetch", action="store_true", help="Do not fetch the model spec after registration")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Configuration file not found: {config_path}")

    payload = {
        "env": args.env,
        "model_id": args.model_id,
        "config": json.loads(config_path.read_text()),
    }

    endpoint = parse.urljoin(args.base_url.rstrip("/") + "/", "registry/configs")
    try:
        response = _post_json(endpoint, payload)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Failed to register configuration ({exc.code}): {detail}")

    model_id = response.get("model_id") or args.model_id
    print("Registered model:", json.dumps(response, indent=2))

    if not args.skip_fetch and model_id:
        spec_url = parse.urljoin(args.base_url.rstrip("/") + "/", f"registry/models/{model_id}")
        try:
            spec = _get_json(spec_url)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SystemExit(f"Failed to fetch model spec ({exc.code}): {detail}")
        print("Fetched stored spec:", json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
