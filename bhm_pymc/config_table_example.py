"""
Tabular Hyperparameter Configuration Example (Master + Engine Tables)

This example shows how to define a HyperParamSpace and run parameters using
two simple configuration tables:
 - Master config table: defines parameter schemas (name, dtype, required, default).
 - Engine config table: provides per-model parameter values (e.g., per env).

We use the in-process OLS model (`linreg_train_and_infer`) and demonstrate
BindingRegistry + inline evaluation and an R-style summary.
"""
from __future__ import annotations

from model_zoo import (
    ModelZoo, ModelRegistry, MetricRegistry,
    ModelSpec, ModelKind, EvalMetricSpec, MetricKind, MetricDirection,
    ExecutionBinding, BindingKind,
    DataRef, DataRefType, ExecutionContext,
    ModelRunRequest, EvaluationSpec,
    InMemoryBindingRegistry, InMemoryRunStore, r_style_summary_string,
)
from model_zoo.runners.library import LibraryModelRunner, LibraryMetricRunner
from model_zoo import DefaultParamValidator
from model_zoo.utils.logging import configure_logging
from model_zoo.utils.config_tables import hyperparam_space_from_master, params_from_engine

# Ensure binding target is importable
from model_zoo.examples.models import linreg_train_and_infer  # noqa: F401
from model_zoo.examples import metrics as example_metrics  # noqa: F401


def main():
    configure_logging("INFO")

    # 1) Define tabular configs (could be loaded from CSV/DB)
    master_rows = [
        {"name": "fit_intercept", "dtype": "bool", "required": False, "default": True, "description": "Include intercept"},
        {"name": "feature_name",  "dtype": "string", "required": False, "default": "X", "description": "Name for coefficient"},
    ]
    engine_rows = [
        {"model_id": "linreg_tabular_v1", "param": "fit_intercept", "value": "true", "env": "dev"},
        {"model_id": "linreg_tabular_v1", "param": "feature_name",  "value": "X",    "env": "dev"},
    ]

    # 2) Build HyperParamSpace and params from tables
    hp_space = hyperparam_space_from_master(master_rows)
    params = params_from_engine(engine_rows, model_id="linreg_tabular_v1", master=hp_space, env="dev")

    # 3) Wire Model Zoo
    model_reg, metric_reg = ModelRegistry(), MetricRegistry()
    runner = LibraryModelRunner()
    runner.set_run_store(InMemoryRunStore())
    bind_reg = InMemoryBindingRegistry()
    zoo = ModelZoo(model_reg, metric_reg, runner, LibraryMetricRunner(), DefaultParamValidator(), binding_registry=bind_reg)

    # 4) Register model + metrics
    model_spec = ModelSpec(
        model_id="linreg_tabular_v1",
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
        bound_id="linreg_tabular_v1", kind=BindingKind.LIBRARY,
        module="model_zoo.examples.models", callable="linreg_train_and_infer",
    )
    r2_binding = ExecutionBinding("r2_v1", BindingKind.LIBRARY, module="model_zoo.examples.metrics", callable="r2_score")
    mse_binding = ExecutionBinding("mse_v1", BindingKind.LIBRARY, module="model_zoo.examples.metrics", callable="mse")
    bind_reg.register_model_binding("linreg_tabular_v1", model_binding)
    bind_reg.register_metric_binding("r2_v1", r2_binding)
    bind_reg.register_metric_binding("mse_v1", mse_binding)

    # 6) Simple synthetic data
    X = [i / 10 for i in range(1, 101)]
    y = [1.0 + 2.0 * x for x in X]
    X_ref = DataRef(DataRefType.INLINE, inline=X)
    y_ref = DataRef(DataRefType.INLINE, inline=y)
    split = 80
    train_idx = list(range(split))
    test_idx = list(range(split, len(X)))

    # 7) Submit run using params from engine table
    req = ModelRunRequest(
        model_id="linreg_tabular_v1",
        inputs={"X": X_ref, "y": y_ref},
        params=params,
        evaluation=EvaluationSpec(
            metric_ids=["r2_v1", "mse_v1"], y_true_key="y",
            train_indices=train_idx, test_indices=test_idx,
            bindings={"r2_v1": r2_binding, "mse_v1": mse_binding},
        ),
    )
    h = zoo.submit_run(req)
    res = zoo.get_run_result(h.run_id)
    print("Tabular-config LinReg status:", res.status.value)
    print(r_style_summary_string(res.summary))


if __name__ == "__main__":
    main()

