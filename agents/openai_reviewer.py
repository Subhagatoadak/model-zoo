"""OpenAI-powered reviewer agent for Model Zoo runs."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except ImportError:  # pragma: no cover - inform users when missing
    OpenAI = None  # type: ignore[assignment]


@dataclass
class OpenAIReviewerAgent:
    """Leverage OpenAI models to provide richer run feedback.

    Args:
        model: OpenAI model identifier (default ``"gpt-5"``).
        api_key: Explicit API key; defaults to ``OPENAI_API_KEY`` environment variable.
        system_instructions: Primer provided to the LLM before analysis.
        enabled: If ``False``, the agent returns heuristic summaries without calling OpenAI.
    """

    model: str = "gpt-5"
    api_key: Optional[str] = None
    system_instructions: str = (
        "You are an experienced ML reviewer. Inspect run diagnostics and evaluation feedback, "
        "highlight issues, explain potential root causes, and suggest next steps. Be concise "
        "but actionable."
    )
    enabled: bool = True

    _client: Optional["OpenAI"] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if OpenAI is None:
            raise ImportError(
                "The 'openai' package is required for OpenAIReviewerAgent. Install it via 'pip install openai'."
            )
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI API key not provided. Set OPENAI_API_KEY or pass api_key to OpenAIReviewerAgent."
            )
        # New SDK: instantiate client once
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

        prompt = self._build_prompt(run_payload, evaluation_summary, config_payload)

        if not self.enabled or self._client is None:
            return {
                "enabled": False,
                "summary": prompt,
                "explanation": (
                    "OpenAI reviewer disabled or not configured; returning prompt context instead of LLM feedback."
                ),
            }

        try:
            response = self._client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": self.system_instructions,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            message = self._extract_output_text(response)
        except Exception as exc:  # pragma: no cover - network call
            message = (
                "Failed to contact OpenAI reviewer. Returning prompt context instead.\n"
                f"Error: {exc}\n\n{prompt}"
            )

        return {
            "enabled": True,
            "model": self.model,
            "review": message,
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_prompt(
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
        parts.append("Provide: 1) Overall assessment; 2) Issues or anomalies; 3) Recommendations.")
        return "\n".join(parts)

    @staticmethod
    def _extract_output_text(response: Any) -> str:
        """Extract combined text from the Responses API payload."""

        # Preferred property in new SDK
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text

        # Fallback: aggregate from output items
        try:
            chunks = []
            for item in getattr(response, "output", []) or []:
                if getattr(item, "type", None) == "message":
                    content = item.message.get("content")
                    if isinstance(content, list):
                        for block in content:
                            piece = block.get("text")
                            if piece:
                                chunks.append(piece)
            if chunks:
                return "\n".join(chunks)
        except Exception:
            pass

        # Legacy compatibility
        try:
            return response.choices[0].message["content"]  # type: ignore[attr-defined]
        except Exception:
            return str(response)
