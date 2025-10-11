"""Tabular Hyperparameter Configuration Example (Master + Engine Tables).

This example shows how to define a HyperParamSpace and run parameters using two
simple configuration tables:
- Master config table: defines parameter schemas (name, dtype, required, default).
- Engine config table: provides per-model parameter values (e.g., per env).

We use the in-process OLS model and demonstrate BindingRegistry, inline
evaluation and an R-style summary.
"""
from __future__ import annotations

import argparse

from model_zoo import (
    ModelZoo, ModelRegistry, MetricRegistry,
    ModelSpec, ModelKind, EvalMetricSpec, MetricKind, MetricDirection,
    ExecutionBinding, BindingKind,
    DataRef, DataRefType, ExecutionContext,
    ModelRunRequest, EvaluationSpec,
    InMemoryBindingRegistry, InMemoryRunStore, InMemoryEventBus,
    r_style_summary_string,
)
from model_zoo.runners.library import LibraryModelRunner, LibraryMetricRunner
from model_zoo import DefaultParamValidator
from model_zoo.utils.logging import configure_logging
import argparse
from model_zoo.utils.config_tables import hyperparam_space_from_master, params_from_engine

# Ensure binding target is importable
# Bindings resolve by module/callable strings; no direct imports needed here.


def main(env: str = "dev", model_id: str = "linreg_tabular_v1"):
    """Demonstrate master/engine config tables for HyperParamSpace and params.

    Includes BindingRegistry resolve, inline evaluation, event bus, and a
    timeout in ExecutionContext. The `env` parameter selects engine rows.
    """
    configure_logging("INFO")

    # 1) Define tabular configs (could be loaded from CSV/DB)
    master_rows = [
        {
            "name": "fit_intercept",
            "dtype": "bool",
            "required": False,
            "default": True,
            "description": "Include intercept",
        },
        {
            "name": "feature_name",
            "dtype": "string",
            "required": False,
            "default": "X",
            "description": "Name for coefficient",
        },
    ]
    engine_rows = [
        {"model_id": model_id, "param": "fit_intercept", "value": "true", "env": env},
        {"model_id": model_id, "param": "feature_name",  "value": "X",    "env": env},
    ]

    # 2) Build HyperParamSpace and params from tables
    try:
        hp_space = hyperparam_space_from_master(master_rows)
        params = params_from_engine(engine_rows, model_id=model_id, master=hp_space, env=env)
    except Exception as e:
        print("[ERROR] invalid config tables:", e)
        return

    # 3) Wire Model Zoo
    model_reg, metric_reg = ModelRegistry(), MetricRegistry()
    runner = LibraryModelRunner(
        max_workers=2,
        max_active_runs=1,
        max_retries=0,
        backoff_initial_s=0.2,
    )
    runner.set_run_store(InMemoryRunStore())
    bind_reg = InMemoryBindingRegistry()
    bus = InMemoryEventBus()
    bus.subscribe("run.status_changed", lambda ev: print("[EVENT] status:", ev.get("status"), ev.get("run_id")))
    bus.subscribe("run.evaluated", lambda ev: print("[EVENT] evaluated:", ev.get("run_id")))
    zoo = ModelZoo(model_reg, metric_reg, runner, LibraryMetricRunner(), DefaultParamValidator(), binding_registry=bind_reg, event_bus=bus)

    # 4) Register model + metrics
    model_spec = ModelSpec(
        model_id=model_id,
        name="LinearRegression_TabularConfig",
        version="1.0.0",
        kind=ModelKind.LIBRARY,
        hyperparams=hp_space,
        tags=["linear", "tabular"],
    )
    zoo.register_model(model_spec)
    r2_spec = EvalMetricSpec("r2_v1", "R2_Score", "1.0.0", MetricDirection.HIGHER_IS_BETTER, MetricKind.LIBRARY)
    mse_spec = EvalMetricSpec("mse_v1", "MeanSquaredError", "1.0.0", MetricDirection.LOWER_IS_BETTER, MetricKind.LIBRARY)
    zoo.register_metric(r2_spec)
    zoo.register_metric(mse_spec)

    # 5) Bindings
    model_binding = ExecutionBinding(
        bound_id=model_id, kind=BindingKind.LIBRARY,
        module="model_zoo.examples.models", callable="linreg_train_and_infer",
    )
    r2_binding = ExecutionBinding("r2_v1", BindingKind.LIBRARY, module="model_zoo.examples.metrics", callable="r2_score")
    mse_binding = ExecutionBinding("mse_v1", BindingKind.LIBRARY, module="model_zoo.examples.metrics", callable="mse")
    bind_reg.register_model_binding(model_id, model_binding)
    bind_reg.register_metric_binding("r2_v1", r2_binding)
    bind_reg.register_metric_binding("mse_v1", mse_binding)

    # 6) Simple synthetic data
    x_vals = [i / 10 for i in range(1, 101)]
    y_vals = [1.0 + 2.0 * x for x in x_vals]
    x_ref = DataRef(DataRefType.INLINE, inline=x_vals)
    y_ref = DataRef(DataRefType.INLINE, inline=y_vals)
    split = 80
    train_idx = list(range(split))
    test_idx = list(range(split, len(x_vals)))

    # 7) Submit run using params from engine table
    req = ModelRunRequest(
        model_id=model_id,
        inputs={"X": x_ref, "y": y_ref},
        params=params,
        evaluation=EvaluationSpec(
            metric_ids=["r2_v1", "mse_v1"], y_true_key="y",
            train_indices=train_idx, test_indices=test_idx,
        ),
        context=ExecutionContext(timeout_s=60),
    )
    h = zoo.submit_run(req)
    res = zoo.get_run_result(h.run_id)
    print("Tabular-config LinReg status:", res.status.value)
    print(r_style_summary_string(res.summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tabular-config Model Zoo example"
    )
    parser.add_argument(
        "--env",
        default="dev",
        help="Engine table environment key (default: dev)",
    )
    parser.add_argument(
        "--model-id",
        default="linreg_tabular_v1",
        help="Target model_id (default: linreg_tabular_v1)",
    )
    args = parser.parse_args()
    main(env=args.env, model_id=args.model_id)
