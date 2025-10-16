"""
Agent utilities for orchestrating Model Zoo workflows.

This package exposes two primary helpers:

- `ModelingAgent`: registers configurations, submits runs, and streams
  results from a Model Zoo API instance.
- `EvaluationAgent`: inspects run outputs and provides structured
  feedback about issues or potential regressions.
"""

from .modeling_agent import ModelingAgent
from .evaluation_agent import EvaluationAgent
from .openai_reviewer import OpenAIReviewerAgent

__all__ = ["ModelingAgent", "EvaluationAgent", "OpenAIReviewerAgent"]
