# How‑Tos

This guide contains concise, step‑by‑step instructions for common tasks in the Model Zoo platform.

## Add a Runner

Goal: Add a concrete runner to execute models (e.g., MCP/HTTP/Pipeline).

1) Create a new runner module

```
src/model_zoo/runners/<your_runner>.py
```

2) Implement the interfaces

- Import and subclass `ModelRunner` and/or `MetricRunner` from `src/model_zoo/runners/base.py`.
- Implement `submit`, `status`, `result`, `cancel` (for models) and `compute` (for metrics).
- Use `ExecutionBinding` (src/model_zoo/bindings/base.py) fields relevant to your transport (e.g., `endpoint` for HTTP, `mcp` for MCP).

3) Add transport‑specific logic

- HTTP: POST submit payload (`ModelRunRequest.to_dict()`), GET status/result; include retries and timeouts.
- MCP: Connect using `MCPBindingOptions` (stdio/streamable_http), invoke the configured tool/prompt with request payload; collect output.
- Pipeline: Create a job in the orchestrator (Prefect/Flyte/Airflow), persist the external run ID, and poll until completion.

4) Integration and events

- Publish lifecycle events through `EventBus` (`run.submitted`, `run.status_changed`, `run.evaluated`).
- Respect `ExecutionContext.timeout_s` and `resources` if applicable.

5) Export your runner in `model_zoo.__init__` if you want a top‑level import.


## Add a New Metric

Goal: Add an evaluation metric computed in‑process via the `LibraryMetricRunner`.

1) Implement a metric callable

```
def my_metric(y_true, y_pred=None, proba=None, samples=None, params=None, extra=None) -> float | dict:
    # return a numeric score, or a dict with keys: score, artifacts, metadata
```

2) Place it in a module importable by bindings

```
src/model_zoo/examples/metrics.py  # or your own package
```

3) Register metric spec and binding

```
from model_zoo import EvalMetricSpec, MetricKind, MetricDirection, ExecutionBinding, BindingKind

metric_spec = EvalMetricSpec(
  metric_id="my_metric_v1", name="MyMetric", version="1.0.0",
  direction=MetricDirection.HIGHER_IS_BETTER, kind=MetricKind.LIBRARY)
zoo.register_metric(metric_spec)

metric_binding = ExecutionBinding(
  bound_id="my_metric_v1", kind=BindingKind.LIBRARY,
  module="path.to.metrics", callable="my_metric")
binding_registry.register_metric_binding("my_metric_v1", metric_binding)
```

4) Use it in an `EvaluationSpec` or compute directly via `LibraryMetricRunner`.


## Wire BindingRegistry

Goal: Keep specs declarative and resolve execution at runtime.

1) Construct and attach a binding registry

```
from model_zoo import InMemoryBindingRegistry
bind_reg = InMemoryBindingRegistry()
zoo = ModelZoo(..., binding_registry=bind_reg)
```

2) Register model and metric bindings

```
bind_reg.register_model_binding("model_id", model_binding)
bind_reg.register_metric_binding("r2_v1", r2_binding)
```

3) Submit without attaching `binding=` on the request — the `ModelZoo` will resolve it.


## Add Agent Selection Logic

Goal: Pick the “best” model for a task/constraints, then run it.

1) Use the built‑in baseline selector

```
from model_zoo.selection.selector import ModelSelectionService, SelectionCriteria

selector = ModelSelectionService(model_registry)
spec = selector.select(SelectionCriteria(kind=ModelKind.LIBRARY, required_tags=["bayesian"]))
```

2) Extend the selector

- Add filters for latency/cost constraints (`metadata`), device support, or dataset modality.
- Incorporate past evaluation results from a persistent store.

3) Resolve binding and run

```
binding = binding_registry.get_model_binding(spec.model_id)
req = ModelRunRequest(model_id=spec.model_id, inputs=..., params=..., evaluation=...)
handle = zoo.submit_run(req)
```


## Deploy to Orchestrators

Goal: Execute long‑running training/inference jobs on systems like Prefect, Flyte, or Airflow.

1) Implement a Pipeline runner

- Create `src/model_zoo/runners/pipeline.py` (or an orchestrator‑specific file).
- Submit jobs using the orchestrator’s SDK; persist external run IDs.
- Implement `status`/`result` by polling the orchestrator and mapping results/artifacts back to `ModelRunResult`.
- Emit lifecycle events via `EventBus`.

2) Bindings for pipeline runs

- Use `ExecutionBinding.pipeline_id` and `ExecutionBinding.orchestrator` to identify the external flow.
- Store orchestrator‑specific options in `ExecutionBinding.options`.

3) Timeouts, cancellation, and retries

- Delegate to orchestrator capabilities (retries, crash recovery, kill signals).
- Keep `ExecutionContext.timeout_s` as a contract; honor it via orchestrator config when possible.

4) Observability

- Export run metrics to Prometheus/OTel.
- Forward orchestrator links/logs in `ModelRunResult.metadata`.

