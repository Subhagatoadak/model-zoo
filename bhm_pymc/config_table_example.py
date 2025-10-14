"""Configuration-driven hyperparameter + variable example with PostgreSQL storage.

The example demonstrates how to:

* Load hyperparameters, runtime parameters, and variable definitions from a
  machine-friendly JSON configuration file.
* Apply per-variable transforms and optional variable grouping prior to
  invoking the model runner.
* Persist run metadata in PostgreSQL rather than the in-memory store.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import quote_plus

from model_zoo import (
    ModelZoo,
    ModelRegistry,
    MetricRegistry,
    ModelSpec,
    ModelKind,
    EvalMetricSpec,
    MetricKind,
    MetricDirection,
    ExecutionBinding,
    BindingKind,
    DataRef,
    DataRefType,
    ExecutionContext,
    ModelRunRequest,
    EvaluationSpec,
    InMemoryBindingRegistry,
    InMemoryEventBus,
    r_style_summary_string,
    DefaultParamValidator,
)
from model_zoo.runners.library import LibraryModelRunner, LibraryMetricRunner
from model_zoo.stores.run_store import PostgresRunStore
from model_zoo.utils.logging import configure_logging
from model_zoo.utils.config_loader import load_model_configuration
from model_zoo.utils.variable_processing import build_variables
from model_zoo.examples.models import make_synthetic_linear


def parse_env_file(path: Path) -> Dict[str, str]:
    """Parse a simple KEY=VALUE env file."""

    values: Dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(f"Env file not found: {path}")
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def build_postgres_dsn(settings: Dict[str, str]) -> Tuple[str, str]:
    """Compose a PostgreSQL DSN from environment key/values."""

    user = settings.get("POSTGRES_USER") or settings.get("POSTGRES_USERNAME")
    password = settings.get("POSTGRES_PASSWORD")
    host = settings.get("POSTGRES_HOST", "localhost")
    port = settings.get("POSTGRES_PORT", "5432")
    database = settings.get("POSTGRES_DB") or settings.get("POSTGRES_DATABASE")
    table = settings.get("POSTGRES_TABLE", "model_runs")

    if not user or not password or not database:
        tuples = [("POSTGRES_USER", user), ("POSTGRES_PASSWORD", password), ("POSTGRES_DB", database)]
        missing = [name for name, value in tuples if not value]
        raise ValueError(f"Missing required PostgreSQL settings: {missing}")

    quoted_user = quote_plus(user)
    quoted_password = quote_plus(password)
    dsn = f"postgresql://{quoted_user}:{quoted_password}@{host}:{port}/{database}"
    return dsn, table


def to_data_ref(value: Any) -> DataRef:
    return DataRef(ref_type=DataRefType.INLINE, inline=value)


def resolve_data_sources(points: int, slope: float, intercept: float, noise: float, config_raw: Dict[str, Any]) -> Dict[str, Any]:
    """Build data sources from synthetic generator and optional config overrides."""

    x_vals, y_vals = make_synthetic_linear(n=points, slope=slope, intercept=intercept, noise_sigma=noise)
    sources: Dict[str, Any] = {"x_values": x_vals, "y_values": y_vals}

    datasets = config_raw.get("datasets") or {}
    for name, spec in datasets.items():
        dtype = (spec.get("type") or "inline").lower()
        if dtype != "inline":
            raise ValueError(f"Unsupported dataset type '{dtype}' for '{name}'")
        sources[name] = spec.get("value")
    return sources


def build_evaluation(cfg_eval, default_train: list[int], default_test: list[int]) -> EvaluationSpec:
    if cfg_eval is None:
        return EvaluationSpec(metric_ids=["r2_v1", "mse_v1"], y_true_key="y", train_indices=default_train, test_indices=default_test)
    return EvaluationSpec(
        metric_ids=cfg_eval.metric_ids,
        y_true_key=cfg_eval.y_true_key,
        compute_train=cfg_eval.compute_train,
        compute_valid=cfg_eval.compute_valid,
        train_indices=cfg_eval.train_indices if cfg_eval.train_indices is not None else default_train,
        test_indices=cfg_eval.test_indices if cfg_eval.test_indices is not None else default_test,
    )


def configure_model_zoo(store: PostgresRunStore) -> ModelZoo:
    model_reg, metric_reg = ModelRegistry(), MetricRegistry()
    runner = LibraryModelRunner(max_workers=2, max_active_runs=1, max_retries=0, backoff_initial_s=0.2)
    runner.set_run_store(store)
    bind_reg = InMemoryBindingRegistry()
    bus = InMemoryEventBus()
    bus.subscribe("run.status_changed", lambda ev: print("[EVENT] status:", ev.get("status"), ev.get("run_id")))
    bus.subscribe("run.evaluated", lambda ev: print("[EVENT] evaluated:", ev.get("run_id")))
    zoo = ModelZoo(model_reg, metric_reg, runner, LibraryMetricRunner(), DefaultParamValidator(), binding_registry=bind_reg, event_bus=bus)
    return zoo


def main(env: str, config_path: str, env_file: str, model_id: str | None, points: int, slope: float, intercept: float, noise: float, train_fraction: float) -> None:
    configure_logging("INFO")

    cfg = load_model_configuration(config_path, env=env, model_id=model_id)

    # Store credentials are sourced from the env file
    env_values = parse_env_file(Path(env_file))
    dsn, table_name = build_postgres_dsn(env_values)

    store = PostgresRunStore(dsn=dsn, table_name=table_name)
    zoo = configure_model_zoo(store)

    # Register model metadata, metrics, and bindings from configuration
    model_info = cfg.model
    model_spec = ModelSpec(
        model_id=model_info["id"],
        name=model_info.get("name", model_info["id"]),
        version=model_info.get("version", "0.0.1"),
        kind=ModelKind[model_info.get("kind", "library").upper()],
        hyperparams=cfg.hyperparam_space,
        tags=model_info.get("tags", []),
    )
    zoo.register_model(model_spec)

    r2_spec = EvalMetricSpec("r2_v1", "R2_Score", "1.0.0", MetricDirection.HIGHER_IS_BETTER, MetricKind.LIBRARY)
    mse_spec = EvalMetricSpec("mse_v1", "MeanSquaredError", "1.0.0", MetricDirection.LOWER_IS_BETTER, MetricKind.LIBRARY)
    zoo.register_metric(r2_spec)
    zoo.register_metric(mse_spec)

    binding_info = model_info.get("binding") or {}
    if not binding_info:
        raise ValueError("Model configuration requires a 'binding' section with module and callable")
    model_binding = ExecutionBinding(bound_id=model_info["id"], kind=BindingKind.LIBRARY, module=binding_info["module"], callable=binding_info["callable"])
    metric_binding_module = "model_zoo.examples.metrics"
    r2_binding = ExecutionBinding("r2_v1", BindingKind.LIBRARY, module=metric_binding_module, callable="r2_score")
    mse_binding = ExecutionBinding("mse_v1", BindingKind.LIBRARY, module=metric_binding_module, callable="mse")
    zoo.binding_registry.register_model_binding(model_info["id"], model_binding)
    zoo.binding_registry.register_metric_binding("r2_v1", r2_binding)
    zoo.binding_registry.register_metric_binding("mse_v1", mse_binding)

    # Prepare data sources and transform variables per configuration
    data_sources = resolve_data_sources(points, slope, intercept, noise, cfg.raw)
    resolved_variables, grouped = build_variables(cfg.variables, data_sources)

    if "X" not in resolved_variables or "y" not in resolved_variables:
        raise ValueError("Configuration must produce variables 'X' and 'y'")

    inputs = {key: to_data_ref(value) for key, value in resolved_variables.items()}

    total = len(resolved_variables["X"])
    split = max(1, min(total, int(total * train_fraction)))
    train_idx = list(range(split))
    test_idx = list(range(split, total))

    params = dict(cfg.params)
    params.setdefault("train_indices", train_idx)

    evaluation = build_evaluation(cfg.evaluation, train_idx, test_idx)

    context_kwargs = dict(cfg.context)
    context = ExecutionContext(**context_kwargs)

    req = ModelRunRequest(
        model_id=model_info["id"],
        inputs=inputs,
        params=params,
        evaluation=evaluation,
        context=context,
    )

    handle = zoo.submit_run(req)
    result = zoo.get_run_result(handle.run_id)
    print("Run status:", result.status.value)
    print(r_style_summary_string(result.summary))
    print("Stored grouped variables:", grouped.keys())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config-driven Model Zoo example")
    parser.add_argument("--env", default="dev", help="Environment key for configuration entries")
    parser.add_argument("--config", default="configs/linreg_config.json", help="Path to the model configuration JSON")
    parser.add_argument("--env-file", default=".env", help="Path to env file with PostgreSQL credentials")
    parser.add_argument("--model-id", default=None, help="Optional override for model id")
    parser.add_argument("--points", type=int, default=100, help="Number of synthetic datapoints")
    parser.add_argument("--slope", type=float, default=2.0, help="Slope for synthetic data generation")
    parser.add_argument("--intercept", type=float, default=1.0, help="Intercept for synthetic data generation")
    parser.add_argument("--noise", type=float, default=0.0, help="Noise sigma for synthetic data generation")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Fraction of samples used for training split")
    args = parser.parse_args()

    main(
        env=args.env,
        config_path=args.config,
        env_file=args.env_file,
        model_id=args.model_id,
        points=args.points,
        slope=args.slope,
        intercept=args.intercept,
        noise=args.noise,
        train_fraction=args.train_fraction,
    )
