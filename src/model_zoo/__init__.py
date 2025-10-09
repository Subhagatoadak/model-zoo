"""Top-level package exports for the Model Zoo.

Importing from here provides convenient access to facade, registries, contracts,
bindings, runners, selection, IO, stores, and utilities.
"""
from .__about__ import __version__

# Facade
from .facade.model_zoo import ModelZoo

# Registries
from .registry.models import ModelSpec, ModelRegistry, ModelKind
from .contracts.hyperparams import HyperParamField, HyperParamSpace, DType
from .registry.metrics import EvalMetricSpec, MetricRegistry, MetricDirection, MetricKind

# Contracts
from .contracts.io import DataRef, DataRefType, SecretRef, ExecutionContext
from .contracts.run_requests import (
    ModelRunRequest, ModelRunResult, MetricComputeRequest, MetricComputeResult, RunStatus, ModelRunHandle, EvaluationSpec
)
from .contracts.capability import ModelCapability, MetricRequirement
from .contracts.validation import ParamValidationService
from .contracts.validators import DefaultParamValidator
from .contracts.audit import AuditLog, AuditEvent

# Bindings & Runners
from .bindings.base import BindingKind, ExecutionBinding
from .bindings.registry import BindingRegistry, InMemoryBindingRegistry
from .bindings.mcp import MCPBindingOptions
from .runners.base import ModelRunner, MetricRunner
from .utils.summary import r_style_summary_string
from .selection.selector import ModelSelectionService
from .io.loaders import DefaultDataLoader
from .stores.run_store import RunStore, InMemoryRunStore
from .events.bus import EventBus, InMemoryEventBus, WebhookEventBus

__all__ = [n for n in dir() if not n.startswith("_")]
