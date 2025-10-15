# model-zoo

Pure registries + contracts + bindings + runner interfaces for a pluggable Model Zoo.

What it gives you
- Registries are declarative only (no execution) for Models and Metrics.
- Bindings tell the platform how to execute models/metrics (API/Library/Pipeline/MCP).
- Runners execute via bindings; includes in‑process Library runners and HTTP stubs.
- Facade (`ModelZoo`) orchestrates registration, validation, runs, and evaluation.

Quick start
1) Install (optionally add extras for sklearn/pymc)

```
pip install -e .
# optional
pip install scikit-learn numpy
pip install pymc numpy
```

2) Run the end‑to‑end example (register + run + evaluate)

```
python bhm_pymc/register_and_run.py
```

This script demonstrates:
- Registering models and metrics.
- Using the in‑process `LibraryModelRunner` and `LibraryMetricRunner`.
- Submitting runs with an EvaluationSpec (train/test split, metric bindings).
- Getting an R‑style summary string with named coefficients and metrics.
- Library models: simple OLS, conjugate Bayesian, sklearn LinearRegression, PyMC NUTS.

Bring your own package
- Package your callable into a wheel (or publish it internally) and declare it in the
  configuration `dependencies`. The registry will install those requirements into a
  cached site-packages directory before importing the binding.
- See `docs/notebooks/custom_model_registration_demo.ipynb` for an end-to-end walkthrough.
- Prefer calling out-of-process APIs instead? Spin up the sample PyMC service (`docker compose up pymc_api`)
  and follow `docs/notebooks/api_pymc_model_demo.ipynb` for registering an `api` binding and running it.

Core concepts
- Model registry: declares a `ModelSpec` with `HyperParamSpace` only.
- Metric registry: declares `EvalMetricSpec` (direction + parameters).
- Bindings: execution description separate from registries, e.g.:
  - Library binding: `{kind: library, module: "pkg.module", callable: "fn"}`
- Runners: platform implementations; use Library runners for in‑process callables.

Programmatic usage (minimal)
```
from model_zoo import (
  ModelZoo, ModelRegistry, MetricRegistry,
  ModelSpec, ModelKind, EvalMetricSpec, MetricKind, MetricDirection,
  HyperParamSpace, HyperParamField, DType,
  ExecutionBinding, BindingKind,
  DataRef, DataRefType, ExecutionContext,
  ModelRunRequest, EvaluationSpec,
  DefaultParamValidator,
)
from model_zoo.runners.library import LibraryModelRunner, LibraryMetricRunner

model_reg, metric_reg = ModelRegistry(), MetricRegistry()
zoo = ModelZoo(model_reg, metric_reg,
               model_runner=LibraryModelRunner(),
               metric_runner=LibraryMetricRunner(),
               validator=DefaultParamValidator())

# Register a model
lin_hp = HyperParamSpace(fields={
  "fit_intercept": HyperParamField("fit_intercept", DType.BOOL, default=True)
})
zoo.register_model(ModelSpec(
  model_id="linear_regression_v1",
  name="LinearRegression_OLS",
  version="1.0.0",
  kind=ModelKind.LIBRARY,
  hyperparams=lin_hp,
))

# Register metrics
zoo.register_metric(EvalMetricSpec(
  metric_id="r2_v1", name="R2_Score", version="1.0.0",
  direction=MetricDirection.HIGHER_IS_BETTER, kind=MetricKind.LIBRARY))
zoo.register_metric(EvalMetricSpec(
  metric_id="mse_v1", name="MeanSquaredError", version="1.0.0",
  direction=MetricDirection.LOWER_IS_BETTER, kind=MetricKind.LIBRARY))

# Bindings
lin_binding = ExecutionBinding(
  bound_id="linear_regression_v1", kind=BindingKind.LIBRARY,
  module="model_zoo.examples.models", callable="linreg_train_and_infer")
r2_binding = ExecutionBinding(
  bound_id="r2_v1", kind=BindingKind.LIBRARY,
  module="model_zoo.examples.metrics", callable="r2_score")
mse_binding = ExecutionBinding(
  bound_id="mse_v1", kind=BindingKind.LIBRARY,
  module="model_zoo.examples.metrics", callable="mse")

# Inputs, request, submit
X = [i/10 for i in range(1, 101)]
y = [1.0 + 2.0*x for x in X]
req = ModelRunRequest(
  model_id="linear_regression_v1",
  inputs={"X": DataRef(DataRefType.INLINE, inline=X),
          "y": DataRef(DataRefType.INLINE, inline=y)},
  params={"fit_intercept": True, "feature_name": "X"},
  binding=lin_binding,
  evaluation=EvaluationSpec(
    metric_ids=["r2_v1", "mse_v1"],
    y_true_key="y",
    train_indices=list(range(80)), test_indices=list(range(80, 100)),
    bindings={"r2_v1": r2_binding, "mse_v1": mse_binding},
  ),
)
handle = zoo.submit_run(req)
# Poll result using your own loop or re‑use the helper in the example script
res = zoo.get_run_result(handle.run_id)
```

Evaluation: train/test + R‑style summary
- Provide an `EvaluationSpec` on the `ModelRunRequest` to compute metrics inline.
- Summary includes Call, named Coefficients, Fit (R²/MSE), Sizes, Metrics (train/validation).
- You can also call `zoo.compute_all_metrics(...)` directly when needed.

Library examples provided
- OLS: `model_zoo.examples.models.linreg_train_and_infer`
- Conjugate Bayesian (Normal/Gamma): `model_zoo.examples.models.bayes_linreg_train_and_infer`
- sklearn LinearRegression: `model_zoo.examples.sklearn_models.sk_lr_train_and_infer`
 - PyMC NUTS (single/multi‑feature): `model_zoo.examples.pymc_models.pymc_linear_nuts`

Priors (Bayesian)
- Conjugate demo:
  - beta: {distribution: normal, mu, tau}
  - sigma: {distribution: gamma|erlang, shape, rate}
- PyMC NUTS:
  - beta: {distribution: normal, mu: scalar|list, tau: scalar|list}
  - sigma: {distribution: half_normal|gamma|erlang, scale|shape|rate}

Feature naming and coefficients
- For single feature: pass `feature_name: "X"`.
- For multiple features: pass `feature_names: [..]` or supply X as `list[dict]` to derive names from keys.
- Summaries use named rows like `coef[wt]`, `coef[hp]`, `intercept`.

Logging and tracing
- Use `model_zoo.utils.logging.configure_logging("INFO")` and `quiet_third_party_loggers()`.
- Each run is tagged with a `trace_id` (attached to result metadata) and audited on status change.

Troubleshooting
- Sklearn import errors or segfaults often indicate binary mismatches between numpy/scipy/sklearn. Use compatible wheel versions (e.g., numpy < 2 for older SciPy/Sklearn) or upgrade SciPy/Sklearn to versions compatible with NumPy 2.x.
- If a run takes long (e.g., PyMC NUTS), ensure your polling waits until status is SUCCEEDED. The example script includes a robust `wait_for_result` with timeout and backoff.

See also
- docs/index.md for architecture and components.
- The example script `bhm_pymc/register_and_run.py` for a complete, runnable reference.
- docs/registry_api.md for running the FastAPI registry service with Docker.

Build and publish (Artifactory)
- Build sdist and wheel using Hatch/PEP 517:

```
python -m pip install --upgrade build twine
python -m build
```

- Upload to Artifactory with twine:

```
export ARTIFACTORY_URL=https://<your-artifactory>/api/pypi/<your-repo>
export TWINE_USERNAME=<username-or-token>
export TWINE_PASSWORD=<password-or-token>
twine upload --repository-url "$ARTIFACTORY_URL" dist/*
```

- Alternatively, use the helper script:

```
ARTIFACTORY_URL=https://<your-artifactory>/api/pypi/<your-repo> \
TWINE_USERNAME=<token> TWINE_PASSWORD=<secret> \
scripts/publish.sh
```
