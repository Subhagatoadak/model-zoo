"""
OpenAI-powered reviewer agent for Model Zoo runs.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]


@dataclass
class OpenAIReviewerAgent:
    """Leverage OpenAI models to provide richer run feedback.

    Args:
        model: OpenAI model identifier (e.g. ``"gpt-4o-mini"``).
        api_key: Explicit API key; defaults to ``OPENAI_API_KEY`` environment variable.
        system_prompt: Primer provided to the LLM before analysis.
        enabled: If ``False``, the agent returns heuristic summaries without calling OpenAI.
    """

    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    system_prompt: str = (
        "You are an experienced ML reviewer. Inspect run diagnostics and evaluation "
        "feedback, highlight issues, explain potential root causes, and suggest next steps. "
        "Be concise but actionable."
    )
    enabled: bool = True

    _client: Optional[Any] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if OpenAI is None:
            raise ImportError(
                "The 'openai' package is required for OpenAIReviewerAgent. "
                "Install it via 'pip install openai'."
            )
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI API key not provided. Set OPENAI_API_KEY or pass api_key to OpenAIReviewerAgent."
            )
        self._client = OpenAI(api_key=api_key)

    # ------------------------------------------------------------------ #
    def review(
        self,
        run_payload: Mapping[str, Any],
        *,
        evaluation_summary: Optional[Mapping[str, Any]] = None,
        config_payload: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Produce a natural-language review of the run outcome."""

        context_block = self._build_context(run_payload, evaluation_summary, config_payload)

        if not self.enabled or self._client is None:
            return {
                "enabled": False,
                "summary": context_block,
                "explanation": (
                    "OpenAI reviewer disabled or not configured; returning raw context summary."
                ),
            }

        try:
            response = self._client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": context_block,
                    },
                ],
            )
            message = self._extract_response_text(response)
        except Exception as exc:  # pragma: no cover - network call
            message = (
                "Failed to contact OpenAI reviewer. Returning context summary instead.\n"
                f"Error: {exc}\n\n{context_block}"
            )

        return {
            "enabled": True,
            "model": self.model,
            "review": message,
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_context(
        run_payload: Mapping[str, Any],
        evaluation_summary: Optional[Mapping[str, Any]],
        config_payload: Optional[Mapping[str, Any]],
    ) -> str:
        parts = ["Run payload:", json.dumps(run_payload, indent=2, default=str)]
        if evaluation_summary is not None:
            parts.append("Evaluation summary:")
            parts.append(json.dumps(evaluation_summary, indent=2, default=str))
        if config_payload is not None:
            minimal_config = {
                "model": config_payload.get("model"),
                "dependencies": config_payload.get("dependencies"),
                "hyperparameters.schema": config_payload.get("hyperparameters", {}).get("schema"),
            }
            parts.append("Relevant configuration excerpt:")
            parts.append(json.dumps(minimal_config, indent=2, default=str))
        parts.append(
            "Provide: 1) Overall assessment; 2) Issues or anomalies; 3) Recommendations."
        )
        return "\n".join(parts)

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """Handle OpenAI Responses API payloads."""

        try:
            for item in response.output:
                if item.type == "message" and item.message.get("content"):
                    return item.message["content"][0].get("text", "") or ""
        except Exception:
            pass
        # Fallback to legacy data structures
        try:
            return response.choices[0].message["content"]  # type: ignore[attr-defined]
        except Exception:
            return str(response)

