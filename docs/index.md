# model-zoo

This library provides:
- **Registries** (pure): Model & Metric specs with machine-fillable hyperparameters.
- **Contracts**: DataRef, Run requests/results, capabilities, validation, audit.
- **Bindings**: How to execute (API/Library/Pipeline/MCP) — kept outside registries.
- **Runners**: Abstract interfaces to execute via bindings. Includes in-process `LibraryModelRunner`/`LibraryMetricRunner` and an HTTP `HttpModelRunner`/`HttpMetricRunner` for API backends.
- **Facade**: ModelZoo to orchestrate without putting code into registries.

Quick Links
- README.md for a runnable quick start and usage patterns
- `bhm_pymc/register_and_run.py` for end-to-end registration + runs + evaluation
- How‑tos: see docs/howtos.md for step‑by‑step guides

Running the examples
1) Install base package and optional extras depending on what you want to run.

```
pip install -e .
# optional
pip install scikit-learn numpy   # for sklearn examples
pip install pymc numpy           # for PyMC NUTS examples
```

2) Execute the demo script

```
python bhm_pymc/register_and_run.py
```

What the demo covers
- Registering OLS, Bayesian conjugate, sklearn LinearRegression, and PyMC NUTS models
- Registering R² and MSE metrics
- Binding models/metrics to in-process callables
- Submitting runs with `EvaluationSpec` to compute train/validation metrics
- Viewing R-style summaries with named coefficients

Configuring priors
- Conjugate demo (normal/gamma): pass `priors` on the ModelRunRequest params
- PyMC NUTS beta prior: normal (scalar or per-feature vectors)
- PyMC NUTS sigma prior: half_normal (scale), gamma/erlang (shape, rate)

Dealing with logs and long runs
- Enable logging: `configure_logging("INFO")`
- Quiet third-party libraries: `quiet_third_party_loggers()`
- Use a robust wait loop to poll `get_run_result` until SUCCEEDED (see demo helper)
