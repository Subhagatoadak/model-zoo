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
    ModelRunRequest, ModelRunResult, MetricComputeRequest, MetricComputeResult, RunStatus, ModelRunHandle
)
from .contracts.capability import ModelCapability, MetricRequirement
from .contracts.validation import ParamValidationService
from .contracts.audit import AuditLog, AuditEvent

# Bindings & Runners
from .bindings.base import BindingKind, ExecutionBinding
from .bindings.mcp import MCPBindingOptions
from .runners.base import ModelRunner, MetricRunner

__all__ = [n for n in dir() if not n.startswith("_")]
