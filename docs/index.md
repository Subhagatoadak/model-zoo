# model-zoo

This library provides:
- **Registries** (pure): Model & Metric specs with machine-fillable hyperparameters.
- **Contracts**: DataRef, Run requests/results, capabilities, validation, audit.
- **Bindings**: How to execute (API/Library/Pipeline/MCP) — kept outside registries.
- **Runners**: Abstract interfaces to execute via bindings. Includes in-process `LibraryModelRunner`/`LibraryMetricRunner` and an HTTP `HttpModelRunner`/`HttpMetricRunner` for API backends.
- **Facade**: ModelZoo to orchestrate without putting code into registries.
