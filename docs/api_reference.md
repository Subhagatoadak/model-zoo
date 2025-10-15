# Model Zoo API Reference

This reference describes each module, class, and method in the Model Zoo, with explanations of what they do and why they exist. Use it alongside the How‑Tos (docs/howtos.md) for step‑by‑step guidance.

## Package Entry: src/model_zoo/__init__.py
- Purpose: Convenience exports for facade, registries, contracts, bindings, runners, selection, IO, stores, events, and utilities. Keeps user imports short and consistent.

## Facade: src/model_zoo/facade/model_zoo.py
- Class `ModelZoo`
  - Why: Central orchestrator that wires registries (pure metadata) to runners (execution) with validation, auditing, and optional event publishing.
  - `register_model(spec, actor="system")` — Add a `ModelSpec` to the registry; emits audit. Prevents mixing metadata with execution code.
  - `register_metric(spec, actor="system")` — Add an `EvalMetricSpec`; emits audit. Keeps metrics declarative.
  - `list_models()` / `list_metrics()` — Discovery helpers for agents/UI.
  - `submit_run(req, actor="system")` — Validates params, resolves binding via `BindingRegistry` if missing, emits `run.submitted`, returns `ModelRunHandle`.
  - `get_run_result(run_id, actor="system")` — Fetches current/final result from runner. Computes inline evaluation (if `EvaluationSpec`) and builds an R‑style summary; emits status/evaluation events.
  - `compute_metric(req, actor="system")` — Validates params and computes a metric via `MetricRunner`.
  - `compute_all_metrics(...)` — Batch convenience for metric computation across registered metrics.
  - Fields: `binding_registry` (resolve execution), `validator` (enforce schemas), `audit` (trace changes), `event_bus` (notify listeners).

## Registries — Models: src/model_zoo/registry/models.py
- Enum `ModelKind` — API, LIBRARY, PIPELINE; why: describe execution mechanism without hard‑wiring the how.
- Dataclass `ModelSpec` — Declarative model entry (id, name, version, kind, tags, metadata, hyperparams). `to_dict()` exports JSON‑friendly representation for audit/transport.
- Class `ModelRegistry` — In‑memory CRUD store for `ModelSpec`. Why: keep metadata organization isolated from execution concerns.

## Registries — Metrics: src/model_zoo/registry/metrics.py
- Enum `MetricDirection` — Optimization direction (higher/lower). Enables correct ranking/selection.
- Enum `MetricKind` — Execution mechanism mirror for metrics.
- Dataclass `EvalMetricSpec` — Declarative metric entry (id, name, version, direction, params schema). `to_dict()` for audit/transport.
- Class `MetricRegistry` — In‑memory CRUD store for `EvalMetricSpec`.

## Contracts — IO: src/model_zoo/contracts/io.py
- Enum `DataRefType` — INLINE or URI. Why: unify IO for runners and callables.
- Dataclass `DataRef` — Reference to data; `to_dict()` for transport. Runners decide how to load URIs.
- Dataclass `SecretRef` — Reference to externally managed secrets (provider/key/version).
- Dataclass `ExecutionContext` — Hints for execution (random seed, timeout, device, resource hints, tags).

## Contracts — Run Requests: src/model_zoo/contracts/run_requests.py
- Enum `RunStatus` — Lifecycle: QUEUED, STARTED, SUCCEEDED, FAILED, CANCELLED.
- Dataclass `EvaluationSpec` — What to evaluate during `get_run_result`: metric_ids, bindings, train/test indices and flags. Why: compute metrics inline and return summaries.
- Dataclass `ModelRunRequest` — Inputs/params/context/dataset_version/secrets, optional binding and evaluation. Why: a complete, transportable execution request.
- Dataclass `ModelRunHandle` — Run id + initial status from `submit`.
- Dataclass `ModelRunResult` — Final/partial result: status, predictions, diagnostics, artifacts, metadata, started/ended timestamps, eval scores, and summary.
- Dataclass `MetricComputeRequest` — Inputs for a metric computation.
- Dataclass `MetricComputeResult` — Score and optional artifacts/metadata.

## Contracts — Hyperparams: src/model_zoo/contracts/hyperparams.py
- Enum `DType` — JSON‑schema base types for hyperparams.
- Dataclass `HyperParamField` — Single param schema (type, required, default, choices/range/pattern); `to_json_schema()` for tooling.
- Dataclass `HyperParamSpace` — Map of fields; `to_json_schema()` for a full object schema.

## Contracts — Validation: src/model_zoo/contracts/validation.py
- Abstract `ParamValidationService` — Interface to validate model/metric params against their `HyperParamSpace`.
- Default implementation: `src/model_zoo/contracts/validators.py` — `DefaultParamValidator` enforces required/type/enum/range/pattern while allowing extra keys. Why: fully declarative schemas without over‑restricting experimentation.

## Contracts — Capability/Audit: src/model_zoo/contracts/capability.py, src/model_zoo/contracts/audit.py
- `ModelCapability` & `MetricRequirement` — Describe production traits and metric IO expectations for compatibility checks. Why: agent can filter/rank models/metrics by capability.
- `CompatibilityService.ensure_compatible(...)` — Placeholder to implement policies.
- `AuditEvent` & `AuditLog` — Structure and interface for audit trails. Why: governance, reproducibility, and compliance.

## Bindings — Base/Registry/MCP
- src/model_zoo/bindings/base.py — `BindingKind` and `ExecutionBinding`; why: encode how to execute models/metrics (module/callable, endpoint, pipeline_id, MCP options).
- src/model_zoo/bindings/registry.py — `BindingRegistry` interface and `InMemoryBindingRegistry`; why: keep registries pure and resolve execution at runtime.
- src/model_zoo/bindings/mcp.py — `MCPBindingOptions` for MCP servers/tools.

## Runners — Base/Library/HTTP/Pipeline/Fake
- src/model_zoo/runners/base.py — Abstract `ModelRunner`/`MetricRunner` interfaces; why: pluggable execution backends.
- src/model_zoo/runners/library.py — In‑process runner: concurrency limits, timeout enforcement, retries/backoff, cooperative cancellation, optional `RunStore`. Why: fast local execution and tests.
- src/model_zoo/runners/http.py — Conventional HTTP implementations for models/metrics. Why: remote services as backends.
- src/model_zoo/runners/pipeline.py — Placeholder for orchestrator‑based runners (Prefect/Flyte/Airflow). Why: long‑running jobs, real cancellation, resource isolation.
- src/model_zoo/runners/fake.py — Minimal fake runners for demos/tests.

## IO — Data Loaders: src/model_zoo/io/loaders.py
- `DefaultDataLoader` — Resolves INLINE DataRefs; returns URI refs unchanged by default. Why: allow user control over IO or pluggable loaders (e.g., file/s3/gs loaders).

## Stores — Runs: src/model_zoo/stores/run_store.py
- `RunStore` interface & `RunRecord` — Persist results for retrieval and lineage.
- `InMemoryRunStore` — Simple in‑memory store; replaceable by DB/cloud stores.

## Events — Bus: src/model_zoo/events/bus.py
- `EventBus` interface — Publish lifecycle events.
- `InMemoryEventBus` — Subscribe/Publish in process; useful for agents/coordinators.
- `WebhookEventBus` — POSTs events to an external endpoint; best‑effort.
- Topics used by ModelZoo: `run.submitted`, `run.status_changed`, `run.evaluated`.

## Selection — Service: src/model_zoo/selection/selector.py
- `SelectionCriteria` — Filter by kind/tags/metadata.
- `ModelSelectionService` — Returns best matching `ModelSpec` (max version). Why: baseline selection; extend with latency/cost/accuracy policies.

## Utilities
- src/model_zoo/utils/logging.py — `configure_logging`, `get_logger`, `quiet_third_party_loggers`; why: consistent logs and noise control (PyMC/ArviZ).
- src/model_zoo/utils/summary.py — `r_style_summary_string(summary)` renders a compact, readable summary from run results.
- src/model_zoo/utils/jsonschema.py — Placeholder for future jsonschema validator integration.
- src/model_zoo/utils/typing.py — Placeholder for project‑wide type aliases.

## Examples — Callables
- src/model_zoo/examples/models.py — OLS linear regression, conjugate Bayesian regression, and synthetic data generators (linear + mtcars‑like).
- src/model_zoo/examples/sklearn_models.py — Sklearn LinearRegression wrapper; flexible inputs; named coefficients.
- src/model_zoo/examples/pymc_models.py — PyMC NUTS callables: linear (per‑feature priors) and hierarchical MMM (geo/brand). Why: demonstrate Bayesian flows with flexible priors.
- src/model_zoo/examples/metrics.py — R2, MSE, MAE metrics for in‑process evaluation.

## Examples — Scripts (bhm_pymc/)
- register_and_run.py — End‑to‑end registration + binding registry + inline evaluation + event bus + timeout/retry/conc.
- marketing_mix_pymc_example.py — Bayesian marketing mix with `coefficient_priors`
  (intercept + variables, constraints) and R-style summaries.
- marketing_mix_pymc_hier_example.py — Hierarchical MMM (geo/brand) with
  `coefficient_priors`, constraints, and R-style summaries.

---

## Design Rationale (Why these concepts exist)
- Pure registries: Keep specifications declarative and portable; allow the platform to decide “how to run”.
- Bindings & BindingRegistry: Decouple execution (library/HTTP/MCP/pipeline) from specs; enable per‑env routing and secure credential handling.
- Runners: Provide pluggable execution backends with consistent interfaces.
- Contracts: Normalize data exchange and lifecycle; enable auditability and reproducibility.
- EvaluationSpec: First‑class evaluation to return metrics/summaries alongside predictions; supports train/test splits.
- Event bus: Let agents subscribe to lifecycle and evaluation events for orchestration and automation.
- Run store: Persist results for later retrieval, lineage, and comparison.
- Selection service: Baseline mechanism for agents to pick models; extend with policies informed by capability and past runs.
