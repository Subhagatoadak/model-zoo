# Model Zoo Architecture

This document summarises the Model Zoo system at two abstraction levels: a high-level solution view (HLD) and a lower-level component design (LLD). It also includes a visual class diagram highlighting the core runtime types.

---

## High-Level Design (HLD)

### Goals
- Provide a registry of ML model specifications, configurations, runs, and diagnostics.
- Allow external clients (CLI, notebooks, automation) to register models and trigger executions.
- Dynamically resolve Python package dependencies for each registered model in an isolated, cacheable environment.
- Offer extensibility for new binding types (library modules, HTTP endpoints, pipelines) and metric evaluations.

### Logical Architecture

```
 ┌──────────────────────────┐
 │      Notebook / CLI      │
 │  (e.g. demos & clients)  │
 └────────────┬─────────────┘
              │HTTP/JSON (REST)
 ┌────────────▼─────────────┐
 │   FastAPI Registry API   │
 │  (`model_zoo.api.app`)   │
 └────────────┬─────────────┘
      Commands│Queries
 ┌────────────▼───────────────┐       ┌────────────────────────┐
 │   RegistryDatabase (PSQL)  │◀──────┤ PostgreSQL (model data)│
 └────────────────────────────┘       └────────────────────────┘
              │
              │In-process requests
 ┌────────────▼───────────────┐
 │    Model Zoo Core Facade   │
 │ (`model_zoo.facade`)       │
 └────────────┬───────────────┘
      submits │runs / evaluates
 ┌────────────▼───────────────┐
 │   LibraryModelRunner       │
 │ + Metric runners           │
 └────────────┬───────────────┘
    loads deps│applies env
 ┌────────────▼───────────────┐
 │  DependencyManager         │
 │  + pip_cache (per model)   │
 └────────────────────────────┘
```

### Key External Interfaces
- **REST API** (FastAPI via uvicorn) for model/spec/config registration, run submission, and status polling.
- **PostgreSQL** backing store for model metadata, configurations, and run tracking.
- **File system** for dependency wheel caches (`~/.model_zoo/dependencies/<hash>`).
- **Optional external services**: third-party APIs invoked via API bindings, metric services.

### Deployment Overview

| Environment | Runtime | Configuration |
|-------------|---------|----------------|
| Local dev   | uvicorn + FastAPI | `.env` file supplies DB credentials and PAT for Azure Artifacts. |
| Container   | `Dockerfile` builds Python 3.11 image, copies repo, sets `MODEL_ZOO_ENV_FILE=/app/.env`. |
| CI/testing  | pytest suite exercises loaders, dependency management, and registry logic with in-memory or temporary DB setups. |

---

## Low-Level Design (LLD)

### API Layer (`model_zoo.api.app`)
- FastAPI app with routers for:
  - `/registry/configs` – register configuration payloads.
  - `/registry/models` – register/list model specs.
  - `/runs` – submit execution requests and poll statuses.
- Performs request validation (Pydantic models), loads configurations, resolves dependencies, and delegates to the facade.
- `_bootstrap_environment()` loads secrets (PAT, DB credentials) from `.env` or `MODEL_ZOO_ENV_FILE` before the app starts.

### Configuration Loading (`model_zoo.utils.config_loader`)
- Parses JSON configuration documents: model metadata, hyperparameter schemas, parameter tables, variables, coefficients, evaluation spec, context, and dependency options.
- `_extract_dependency_options()` interprets `dependency_options` block, normalises index URLs, resolves environment variables (from `os.environ` and `.env` files), and returns `DependencyOptions`.
- Produces a `ModelConfiguration` data object consumed by downstream components.

### Dependency Management (`model_zoo.utils.dependency_manager`)
- `DependencyManager.ensure()` normalises requirements, builds a hash key, installs packages into a cached target directory via pip, and writes a `pip_install.json` log.
- Supports additional pip flags (`--index-url`, `--extra-index-url`, `--trusted-host`) with credentials resolved from environment variables or `.env`.
- Sanitises logs to avoid leaking credentials and exposes `DependencyEnvironment` with import paths for runtime.

### Registry & Facade (`model_zoo.registry`, `model_zoo.facade`)
- `RegistryDatabase` encapsulates PostgreSQL access for specs/configs.
- `ModelRegistry`, `MetricRegistry` maintain in-memory copies for quick lookup and binding resolution.
- `ModelZoo` orchestrates model runs, bridging between registries, runners, validators, and dependency/application state.

### Runners (`model_zoo.runners`)
- `LibraryModelRunner` executes Python callables in a thread pool, handles concurrency, cancellation, retries.
- Binds to library execution via `ExecutionBinding` (module/callable), optionally attaches dependency import paths from `DependencyManager`.
- Metric runners (e.g., `LibraryMetricRunner`) evaluate metrics post-execution using similar binding mechanisms.

### Background Execution & Stores
- `InMemoryRunStore` keeps run metadata for quick retrieval.
- Run submission returns an async handle; status polling exposes success/failure, predictions, diagnostics, artifacts, summaries.

### Dependency Resolution Flow
1. Client registers config (JSON) with dependency requirements + PAT reference.
2. API loads configuration, resolves PAT via environment or `.env`.
3. `DependencyManager.ensure()` installs dependencies into a hashed directory.
4. Binding registry records module/callable mapping and augments sys.path for future runs.


---

## Class Diagram (visual)

```
                        ┌───────────────────────────────────┐
                        │ ModelConfiguration                │
                        │───────────────────────────────────│
                        │ + raw: Dict[str, Any]             │
                        │ + model: Dict[str, Any]           │
                        │ + hyperparam_space: HyperParam... │
                        │ + params: Dict[str, Any]          │
                        │ + variables: VariableConfig       │
                        │ + coefficients: CoefficientConfig │
                        │ + evaluation: Optional[...]       │
                        │ + context: Dict[str, Any]         │
                        │ + dependencies: List[str]         │
                        │ + dependency_options: Dependency..│
                        └──────────────▲────────────────────┘
                                       │created by
                                       │
┌──────────────────────────────────────┴─────────────────────────────────┐
│ Config Loader Helpers (module)                                         │
│────────────────────────────────────────────────────────────────────────│
│ load_model_configuration*                                              │
│  └─ _load_model_configuration_from_payload                             │
│         ├─ _extract_variables                                          │
│         ├─ _extract_coefficients                                       │
│         ├─ _extract_dependencies                                       │
│         └─ _extract_dependency_options                                 │
└──────────────────────────────────────┬─────────────────────────────────┘
                                       │produces configs for
                                       │
┌──────────────────────────────────────▼─────────────────────────────────┐
│ ModelZoo (facade)                                                       │
│────────────────────────────────────────────────────────────────────────│
│ + submit_run(request: ModelRunRequest)                                 │
│ + get_run_result(run_id: str)                                          │
│ + cancel_run(run_id: str)                                              │
│ - model_registry: ModelRegistry                                        │
│ - metric_registry: MetricRegistry                                      │
│ - model_runner: ModelRunner                                            │
│ - metric_runner: MetricRunner                                          │
│ - dependency_manager: DependencyManager                                │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │delegates library executions to
                            │
┌───────────────────────────▼─────────────────────────────────────────────┐
│ LibraryModelRunner                                                      │
│────────────────────────────────────────────────────────────────────────│
│ + submit(req: ModelRunRequest) → ModelRunHandle                         │
│ + result(run_id: str) → ModelRunResult                                  │
│ + cancel(run_id: str)                                                   │
│ - dependency_manager: DependencyManager                                 │
│ - binding_registry: Binding registry                                    │
│ - run_store: Optional[RunStore]                                         │
└────────────────────┬────────────────────────────────────────────────────┘
                     │requires dependency env from
                     │
┌────────────────────▼────────────────────────────────────────────────────┐
│ DependencyManager                                                       │
│────────────────────────────────────────────────────────────────────────│
│ + ensure(deps: Iterable[str], options: DependencyOptions)               │
│      → Optional[DependencyEnvironment]                                  │
│ + apply(env: DependencyEnvironment)                                     │
│ - base_dir: Path                                                        │
│ - normalize()                                                           │
│ - pip installation w/ logging                                           │
│                                                                        │
│   ┌────────────────────────────┐                                        │
│   │ DependencyEnvironment      │                                        │
│   │────────────────────────────│                                        │
│   │ + key: str                 │                                        │
│   │ + location: Path           │                                        │
│   │ + requirements: Sequence[str]                                       │
│   │ + import_paths → List[str] (property)                               │
│   └────────────────────────────┘                                        │
└────────────────────────────────────────────────────────────────────────┘
                                         ▲
                                         │embedded in
                                         │
┌────────────────────────────────────────▼────────────────────────────────┐
│ ExecutionBinding                                                         │
│─────────────────────────────────────────────────────────────────────────│
│ + bound_id: str                                                          │
│ + kind: BindingKind (library/api/pipeline)                               │
│ + module: Optional[str]                                                  │
│ + callable: Optional[str]                                                │
│ + endpoint/method/options (for API)                                     │
│ + dependencies: Tuple[str, ...]                                          │
│ + extra_paths: List[str]                                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Class Dictionary

| Class / Function | Key Attributes & Methods | Notes |
|------------------|--------------------------|-------|
| `ModelConfiguration` | `raw`, `model`, `hyperparam_space`, `params`, `variables`, `coefficients`, `evaluation`, `context`, `dependencies`, `dependency_options` | Immutable representation of a loaded JSON config, consumed by API/facade. |
| `load_model_configuration`, `_extract_*` helpers | Extract sections from JSON, enforce env filtering, resolve dependency options (including env variables and PAT). | Produce `ModelConfiguration`. |
| `ModelZoo` | `submit_run`, `get_run_result`, `cancel_run`; holds registries, runners, dependency manager. | Facade orchestrating execution and evaluation. |
| `LibraryModelRunner` | `submit`, `result`, `cancel`; manages thread pool, run store, bindings. | Executes library bindings; interacts with dependency manager. |
| `DependencyManager` | `ensure`, `apply`, `normalize`; has `DependencyEnvironment`. | Resolves package installs, caches pip environments, writes logs. |
| `DependencyEnvironment` | `key`, `location`, `requirements`, `import_paths`. | Encapsulates a prepared dependency directory. |
| `ExecutionBinding` | `bound_id`, `kind`, `module`, `callable`, `endpoint`, `method`, `options`, `dependencies`, `extra_paths`. | Describes how a model/metric should be executed (library vs API). |

---

## Visual LLD & End-to-End Workflow

```
STEP 1 ─ REGISTER CONFIGURATION
┌──────────────────────────┐      POST /registry/configs       ┌────────────────────────────┐
│ Client (Notebook / CLI)  │ ────────────────────────────────▶ │ FastAPI Config Endpoint    │
│  - build config_payload  │                                   │  - _bootstrap_environment  │
│  - include dependency    │                                   │  - load_model_configuration│
│    options (PAT, etc.)   │                                   │  - register ExecutionBinding
└─────────────┬────────────┘                                   └─────────────┬──────────────┘
              │ ModelConfiguration                                     │
              ▼                                                        ▼
     ┌────────────────────────┐                         ┌────────────────────────────┐
     │ Config Loader Helpers  │ ─ produce ───────────▶ │ DependencyManager.ensure    │
     │  _extract_* sections   │                         │  - resolve PAT / env       │
     │  validate JSON schema  │                         │  - pip install cached deps │
     └────────────┬───────────┘                         └────────────┬──────────────┘
                  │                                                │
                  ▼                                                ▼
         ┌─────────────────┐                            ┌─────────────────────────┐
         │ ModelRegistry   │◀──── upsert spec ───────── │ BindingRegistry         │
         └─────────────────┘                            │  - stores ExecutionBinding
                                                       │  - associates dependency env
                                                       └─────────────────────────┘

STEP 2 ─ SUBMIT RUN
┌──────────────────────────┐      POST /runs                ┌────────────────────────────┐
│ Client (Notebook / CLI)  │ ─────────────────────────────▶ │ FastAPI Run Endpoint       │
│  - provide data refs     │                                │  - fetch stored config     │
│  - set params/context    │                                │  - load_resulting config   │
└─────────────┬────────────┘                                │  - build ModelRunRequest   │
              │ ModelRunRequest                            └─────────────┬──────────────┘
              ▼                                                        │
        ┌────────────────────────┐                                     ▼
        │ ModelZoo Facade        │ ─ orchestrates ─────────────────▶ ┌────────────────────────┐
        │  - lookup binding      │                                 │ LibraryModelRunner      │
        │  - ensure dependencies │                                 │  - apply dependency env │
        │  - submit to runner    │                                 │  - execute callable     │
        └────────────┬───────────┘                                 │  - persist run state    │
                     │                                             └────────────┬────────────┘
                     ▼                                                         │
             ┌──────────────┐                                                  ▼
             │ Run Store    │◀─────────── save status/result ───────────────┐
             └──────────────┘                                                │
                                                                             │
STEP 3 ─ POLL RESULTS                                                         │
┌──────────────────────────┐  GET /runs/{id}     ┌──────────────────────────┐│
│ Client                   │ ─────────────────▶ │ FastAPI Run Status        ││
│  - poll for completion   │                   │  - fetch run record       ││
│  - retrieve output       │                   │  - format response        ││
└──────────────────────────┘                   └──────────────┬────────────┘│
                                                             ▼             │
                                          ┌──────────────────────────────┐  │
                                          │ Response Payload             │◀─┘
                                          │  - predictions               │
                                          │  - diagnostics               │
                                          │  - artifacts (URIs/inline)   │
                                          │  - summary text / metrics    │
                                          └──────────────────────────────┘
```

This unified diagram highlights the critical components (LD-level classes/modules) and the end-to-end flow—from configuration registration, through dependency preparation and execution, to result retrieval.

---

## API-Based Model Registry

The API also supports an API-based registry model binding, allowing model execution requests to be proxied to external services instead of local Python callables.

### Workflow
1. Configuration declares a binding with `kind: "api"`, providing `endpoint`, optional `method` and `options`.
2. On registration, `ExecutionBinding` stores the HTTP target instead of module/callable.
3. During run submission, `ModelZoo` detects API bindings and calls `_perform_api_binding_run` (`model_zoo.api.app`). Inputs, parameters, and context are serialised and forwarded over HTTP.
4. Responses are validated for JSON payloads containing `predictions`, `diagnostics`, `artifacts`, etc.

### Architecture Placement
```
┌──────────────┐
│ Model Config │ (binding.kind = "api")
└──────┬───────┘
       │ registers ExecutionBinding(kind=API)
┌──────▼────────┐
│ BindingRegistry│
└──────┬────────┘
       │ run submission detects API binding
┌──────▼────────┐
│ API Adapter   │ (_perform_api_binding_run)
│  - Serialise inputs/params/context
│  - HTTP request to external service
│  - Map response into RunResult        External API
└──────┬────────┘                       ┌───────────────┐
       │ run status references          │ Model Service │
       └───────────────────────────────▶│ (user-defined)│
                                        └───────────────┘
```

This allows the registry to act as a lightweight orchestrator while delegating actual computation to remote prediction services, A/B testing gateways, or legacy systems.


---

## Operational Checklist

1. **Environment**: place credentials in `.env` (including `AZURE_ARTIFACTS_PAT`). For Docker, it is copied to `/app/.env` and used via `MODEL_ZOO_ENV_FILE`.
2. **Bootstrap**: API startup calls `_bootstrap_environment()` to ingest `.env` before resolving dependencies.
3. **Register Model**: Use notebook or CLI to POST configuration; inspect `pip_install.json` on failures.
4. **Submit Run**: POST to `/runs`, then poll `/runs/{id}` for results, diagnostics, and artifacts.
5. **Cleanup**: Dependency caches live under `~/.model_zoo/dependencies/<hash>` and are reused between runs to minimise reinstalls.

This layered architecture isolates configuration parsing, dependency resolution, execution, and persistence concerns, enabling modular extension for new model bindings, metrics, and deployment topologies.
