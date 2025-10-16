"""
Agents for interacting with the Model Zoo API.

The `ModelingAgent` encapsulates the steps of registering a model
configuration, submitting runs, monitoring execution, and optionally
passing results to an evaluation agent.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Mapping

import requests


class ModelingAgentError(RuntimeError):
    """Raised when the modeling agent cannot complete a request."""


@dataclass
class ModelingAgent:
    """High-level helper for running models through the Model Zoo API.

    Args:
        base_url: Root URL of the Model Zoo API (e.g. ``http://localhost:8000``).
        session: Optional requests session for connection reuse/auth.
        poll_interval: Seconds between run status polls.
        run_timeout: Maximum seconds to wait for a run to complete before aborting.
    """

    base_url: str = "http://localhost:8000"
    session: requests.Session | None = None
    poll_interval: float = 1.0
    run_timeout: float = 300.0

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()

    # ------------------------------------------------------------------ #
    # Configuration registration
    # ------------------------------------------------------------------ #
    def register_configuration(
        self,
        config_payload: Mapping[str, Any],
        env: Optional[str] = None,
        model_id_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a configuration document with the Model Zoo API."""

        endpoint = f"{self.base_url.rstrip('/')}/registry/configs"
        body: Dict[str, Any] = {"config": config_payload}
        if env is not None:
            body["env"] = env
        if model_id_override is not None:
            body["model_id"] = model_id_override

        response = self.session.post(endpoint, json=body, timeout=60)
        if not response.ok:
            self._raise_api_error(response, "configuration registration failed")
        return response.json()

    # ------------------------------------------------------------------ #
    # Run submission and monitoring
    # ------------------------------------------------------------------ #
    def submit_run(
        self,
        model_id: str,
        data_sources: Mapping[str, Any],
        env: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Submit a run request and block until completion or timeout."""

        submit_endpoint = f"{self.base_url.rstrip('/')}/runs"
        payload: Dict[str, Any] = {
            "model_id": model_id,
            "data_sources": data_sources,
        }
        if env is not None:
            payload["env"] = env
        if params:
            payload["params"] = dict(params)
        if context:
            payload["context"] = dict(context)

        response = self.session.post(submit_endpoint, json=payload, timeout=60)
        if not response.ok:
            self._raise_api_error(response, "run submission failed")
        ticket = response.json()
        run_id = ticket.get("run_id")
        if not run_id:
            raise ModelingAgentError("Run submission did not return a run_id")

        return self._poll_run_status(run_id)

    def _poll_run_status(self, run_id: str) -> Dict[str, Any]:
        """Poll the run status endpoint until completion."""

        status_endpoint = f"{self.base_url.rstrip('/')}/runs/{run_id}"
        deadline = time.monotonic() + float(self.run_timeout)
        while True:
            response = self.session.get(status_endpoint, timeout=30)
            if not response.ok:
                self._raise_api_error(response, f"failed to fetch status for run {run_id}")
            payload = response.json()
            status = payload.get("status", "").lower()
            if status in {"succeeded", "failed", "cancelled"}:
                return payload
            if time.monotonic() > deadline:
                raise ModelingAgentError(
                    f"Run {run_id} did not complete within {self.run_timeout} seconds"
                )
            time.sleep(self.poll_interval)

    # ------------------------------------------------------------------ #
    # Combined workflow
    # ------------------------------------------------------------------ #
    def execute_workflow(
        self,
        config_payload: Mapping[str, Any],
        data_sources: Mapping[str, Any],
        env: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        context: Optional[Mapping[str, Any]] = None,
        evaluation_agent: "EvaluationAgent | None" = None,
        reviewer_agent: "OpenAIReviewerAgent | None" = None,
    ) -> Dict[str, Any]:
        """Register configuration, run the model, and optionally evaluate."""

        registration = self.register_configuration(config_payload, env=env)
        model_id = registration["model_id"]

        run_result = self.submit_run(
            model_id=model_id,
            data_sources=data_sources,
            env=env,
            params=params,
            context=context,
        )

        evaluation_result: Optional[Dict[str, Any]] = None
        if evaluation_agent is not None:
            evaluation = evaluation_agent.evaluate(run_result, config_payload=config_payload)
            evaluation_result = evaluation
            run_result.setdefault("evaluation_feedback", evaluation)

        if reviewer_agent is not None:
            review = reviewer_agent.review(
                run_payload=run_result,
                evaluation_summary=evaluation_result,
                config_payload=config_payload,
            )
            run_result.setdefault("review_feedback", review)

        return run_result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _raise_api_error(response: requests.Response, message: str) -> None:
        """Raise a detailed error using API response content."""

        detail: Optional[str]
        try:
            parsed = response.json()
            detail = json.dumps(parsed, indent=2)
        except ValueError:
            detail = response.text

        raise ModelingAgentError(f"{message}: HTTP {response.status_code}\n{detail}")


# Avoid circular import during type-checking/runtime.
from .evaluation_agent import EvaluationAgent  # noqa: E402  # isort:skip

from .openai_reviewer import OpenAIReviewerAgent  # noqa: E402  # isort:skip
