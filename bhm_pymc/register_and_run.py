"""Config-driven registration and execution example for linear models."""
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Tuple

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
from model_zoo.contracts.audit import AuditLog, AuditEvent
from model_zoo.runners.library import LibraryModelRunner, LibraryMetricRunner
from model_zoo.stores.run_store import PostgresRunStore
from model_zoo.utils.config_loader import load_model_configuration, ModelConfiguration
from model_zoo.utils.variable_processing import build_variables
from model_zoo.utils.env import read_env_file, build_postgres_dsn
from model_zoo.utils.logging import configure_logging, quiet_third_party_loggers, get_logger
from model_zoo.examples.models import make_synthetic_linear


class PrintAudit(AuditLog):
    def append(self, event: AuditEvent) -> None:
        print(f"[AUDIT] {event.kind} ref={event.ref_id}")


def configure_zoo(dsn: str, table: str) -> Tuple[ModelZoo, InMemoryBindingRegistry]:
    model_reg, metric_reg = ModelRegistry(), MetricRegistry()
    run_store = PostgresRunStore(dsn=dsn, table_name=table)
    model_runner = LibraryModelRunner(max_workers=4, max_active_runs=2, max_retries=1, backoff_initial_s=0.2)
    model_runner.set_run_store(run_store)
    metric_runner = LibraryMetricRunner()
    bind_reg = InMemoryBindingRegistry()
    bus = InMemoryEventBus()
    bus.subscribe("run.status_changed", lambda ev: print("[EVENT] status:", ev.get("status"), ev.get("run_id")))
    bus.subscribe("run.evaluated", lambda ev: print("[EVENT] evaluated:", ev.get("run_id")))
    zoo = ModelZoo(
        model_registry=model_reg,
        metric_registry=metric_reg,
        model_runner=model_runner,
        metric_runner=metric_runner,
        validator=DefaultParamValidator(),
        binding_registry=bind_reg,
        audit=PrintAudit(),
        event_bus=bus,
    )
    return zoo, bind_reg


def register_metrics(zoo: ModelZoo) -> None:
    r2_spec = EvalMetricSpec("r2_v1", "R2_Score", "1.0.0", MetricDirection.HIGHER_IS_BETTER, MetricKind.LIBRARY)
    mse_spec = EvalMetricSpec("mse_v1", "MeanSquaredError", "1.0.0", MetricDirection.LOWER_IS_BETTER, MetricKind.LIBRARY)
    zoo.register_metric(r2_spec, actor="example")
    zoo.register_metric(mse_spec, actor="example")


def register_metric_bindings(bind_reg: InMemoryBindingRegistry) -> None:
    r2_binding = ExecutionBinding("r2_v1", BindingKind.LIBRARY, module="model_zoo.examples.metrics", callable="r2_score")
    mse_binding = ExecutionBinding("mse_v1", BindingKind.LIBRARY, module="model_zoo.examples.metrics", callable="mse")
    bind_reg.register_metric_binding("r2_v1", r2_binding)
    bind_reg.register_metric_binding("mse_v1", mse_binding)


def register_model_from_config(zoo: ModelZoo, bind_reg: InMemoryBindingRegistry, cfg: ModelConfiguration) -> None:
    model_info = cfg.model
    model_spec = ModelSpec(
        model_id=model_info["id"],
        name=model_info.get("name", model_info["id"]),
        version=model_info.get("version", "0.0.1"),
        kind=ModelKind[model_info.get("kind", "library").upper()],
        hyperparams=cfg.hyperparam_space,
        tags=model_info.get("tags", []),
    )
    zoo.register_model(model_spec, actor="example")

    binding_info = model_info.get("binding") or {}
    binding = ExecutionBinding(
        bound_id=model_info["id"],
        kind=BindingKind.LIBRARY,
        module=binding_info["module"],
        callable=binding_info["callable"],
    )
    bind_reg.register_model_binding(model_info["id"], binding)


def build_data_sources(x_vals: List[float], y_vals: List[float]) -> Dict[str, List[float]]:
    return {
        "x_values": x_vals,
        "y_values": y_vals,
    }


def build_inputs(cfg: ModelConfiguration, data_sources: Dict[str, List[float]]) -> Tuple[Dict[str, DataRef], Dict[str, Any]]:
    variables, groups = build_variables(cfg.variables, data_sources)
    input_keys = cfg.model.get("inputs", ["X", "y"])
    inputs: Dict[str, DataRef] = {}
    for key in input_keys:
        value = variables.get(key)
        if value is None:
            raise KeyError(f"Configuration inputs missing key '{key}'")
        inputs[key] = DataRef(ref_type=DataRefType.INLINE, inline=value)
    return inputs, groups


def build_evaluation(cfg: ModelConfiguration, train_idx: List[int], test_idx: List[int]) -> EvaluationSpec:
    if cfg.evaluation is None:
        return EvaluationSpec(metric_ids=["r2_v1", "mse_v1"], y_true_key="y", train_indices=train_idx, test_indices=test_idx)
    spec = cfg.evaluation
    return EvaluationSpec(
        metric_ids=spec.metric_ids,
        y_true_key=spec.y_true_key,
        compute_train=spec.compute_train,
        compute_valid=spec.compute_valid,
        train_indices=spec.train_indices if spec.train_indices is not None else train_idx,
        test_indices=spec.test_indices if spec.test_indices is not None else test_idx,
    )


def submit_run(zoo: ModelZoo, cfg: ModelConfiguration, inputs: Dict[str, DataRef], train_idx: List[int], test_idx: List[int]) -> str:
    params = dict(cfg.params)
    params.setdefault("train_indices", train_idx)
    context = ExecutionContext(**cfg.context)
    evaluation = build_evaluation(cfg, train_idx, test_idx)
    req = ModelRunRequest(
        model_id=cfg.model["id"],
        inputs=inputs,
        params=params,
        context=context,
        evaluation=evaluation,
    )
    handle = zoo.submit_run(req)
    return handle.run_id


def wait_for_result(zoo: ModelZoo, run_id: str, actor: str = "example", timeout_s: float = 600.0):
    import time
    start, delay = time.monotonic(), 0.05
    while True:
        res = zoo.get_run_result(run_id, actor=actor)
        if res.status.name in ("SUCCEEDED", "FAILED", "CANCELLED"):
            return res
        if not res.metadata.get("in_progress"):
            return res
        if time.monotonic() - start > timeout_s:
            return res
        time.sleep(delay)
        delay = min(delay * 1.5, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Config-driven register and run example")
    parser.add_argument("--env", default="dev", help="Environment key for configuration entries")
    parser.add_argument("--env-file", default=".env", help="Path to env file with PostgreSQL credentials")
    parser.add_argument("--linear-config", default="configs/linear_regression.json", help="Path to linear model config")
    parser.add_argument("--bayes-config", default="configs/bayes_linear.json", help="Path to bayesian model config")
    parser.add_argument("--points", type=int, default=100, help="Number of synthetic datapoints")
    parser.add_argument("--slope", type=float, default=2.0, help="Slope for synthetic data generation")
    parser.add_argument("--intercept", type=float, default=1.0, help="Intercept for synthetic data")
    parser.add_argument("--noise", type=float, default=0.5, help="Noise sigma for synthetic data")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Fraction of samples used for training split")
    args = parser.parse_args()

    configure_logging("INFO")
    quiet_third_party_loggers()
    log = get_logger(__name__)
    log.info("Starting config-driven register and run example")

    env_settings = read_env_file(args.env_file)
    dsn, table_name = build_postgres_dsn(env_settings)
    zoo, bind_reg = configure_zoo(dsn, table_name)
    register_metrics(zoo)
    register_metric_bindings(bind_reg)

    linear_cfg = load_model_configuration(args.linear_config, env=args.env)
    bayes_cfg = load_model_configuration(args.bayes_config, env=args.env)
    register_model_from_config(zoo, bind_reg, linear_cfg)
    register_model_from_config(zoo, bind_reg, bayes_cfg)

    X, y = make_synthetic_linear(n=args.points, slope=args.slope, intercept=args.intercept, noise_sigma=args.noise)
    train_split = max(1, min(len(X), int(len(X) * args.train_fraction)))
    train_idx = list(range(train_split))
    test_idx = list(range(train_split, len(X)))

    data_sources = build_data_sources(X, y)
    linear_inputs, linear_groups = build_inputs(linear_cfg, data_sources)
    bayes_inputs, bayes_groups = build_inputs(bayes_cfg, data_sources)

    linear_run_id = submit_run(zoo, linear_cfg, linear_inputs, train_idx, test_idx)
    bayes_run_id = submit_run(zoo, bayes_cfg, bayes_inputs, train_idx, test_idx)

    linear_res = wait_for_result(zoo, linear_run_id, actor="linear")
    bayes_res = wait_for_result(zoo, bayes_run_id, actor="bayes")

    print("Linear status:", linear_res.status.value)
    print(r_style_summary_string(linear_res.summary))
    print("Linear groups:", list(linear_groups.keys()))
    print("Bayesian status:", bayes_res.status.value)
    print(r_style_summary_string(bayes_res.summary))
    print("Bayesian groups:", list(bayes_groups.keys()))


if __name__ == "__main__":
    main()
