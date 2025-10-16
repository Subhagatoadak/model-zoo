"""
Evaluation agent that inspects Model Zoo run results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class EvaluationAgent:
    """Analyse Model Zoo run outputs and surface potential issues.

    Args:
        expected_status: Desired terminal status (default: ``"succeeded"``).
        metric_thresholds: Mapping of metric name to minimum acceptable value.
        required_artifacts: Artifacts that must be present in the result payload.
    """

    expected_status: str = "succeeded"
    metric_thresholds: Mapping[str, float] = field(default_factory=dict)
    required_artifacts: List[str] = field(default_factory=list)

    def evaluate(
        self,
        run_payload: Mapping[str, Any],
        *,
        config_payload: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return an evaluation report for the given run payload."""

        issues: List[str] = []
        recommendations: List[str] = []

        status = str(run_payload.get("status") or "").lower()
        if status != self.expected_status.lower():
            issues.append(
                f"Run completed with status '{status}' (expected '{self.expected_status}')"
            )

        diagnostics = run_payload.get("diagnostics") or {}
        metrics = diagnostics.get("metrics") or {}

        for metric, threshold in self.metric_thresholds.items():
            observed = self._lookup_metric(metrics, metric)
            if observed is None:
                issues.append(f"Missing metric '{metric}' in diagnostics")
            elif observed < threshold:
                issues.append(
                    f"Metric '{metric}'={observed:.4f} below threshold {threshold:.4f}"
                )

        artifacts = run_payload.get("artifacts") or {}
        missing_artifacts = [name for name in self.required_artifacts if name not in artifacts]
        if missing_artifacts:
            issues.append(
                f"Missing expected artifacts: {', '.join(sorted(missing_artifacts))}"
            )

        if config_payload:
            run_inputs = run_payload.get("inputs") or {}
            expected_inputs = config_payload.get("model", {}).get("inputs", [])
            for name in expected_inputs:
                if name not in run_inputs:
                    recommendations.append(
                        f"Input '{name}' specified in config was not present in run payload"
                    )

        summary = {
            "status": status,
            "issues": issues,
            "recommendations": recommendations,
            "metrics": metrics,
            "artifacts": list(artifacts.keys()),
        }
        summary["passed"] = not issues
        return summary

    @staticmethod
    def _lookup_metric(metrics: Mapping[str, Any], metric: str) -> Optional[float]:
        """Attempt to locate a numeric metric value by key."""

        if metric in metrics and isinstance(metrics[metric], (int, float)):
            return float(metrics[metric])
        for value in metrics.values():
            if isinstance(value, Mapping) and metric in value:
                sub_value = value[metric]
                if isinstance(sub_value, (int, float)):
                    return float(sub_value)
        return None

