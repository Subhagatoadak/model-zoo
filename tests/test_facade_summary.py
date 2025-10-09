from model_zoo import (
    ModelZoo, ModelRegistry, MetricRegistry,
    ModelSpec, ModelKind, EvalMetricSpec, MetricKind, MetricDirection,
    HyperParamSpace, HyperParamField, DType,
    ExecutionBinding, BindingKind,
    DataRef, DataRefType,
    ModelRunRequest, EvaluationSpec,
    DefaultParamValidator,
)
from model_zoo.runners.library import LibraryModelRunner, LibraryMetricRunner


def test_facade_run_with_evaluation_and_summary():
    model_reg, metric_reg = ModelRegistry(), MetricRegistry()
    zoo = ModelZoo(
        model_reg, metric_reg,
        model_runner=LibraryModelRunner(), metric_runner=LibraryMetricRunner(),
        validator=DefaultParamValidator(),
    )

    # Register model
    lin_hp = HyperParamSpace(fields={
        "fit_intercept": HyperParamField("fit_intercept", DType.BOOL, default=True)
    })
    zoo.register_model(ModelSpec(
        model_id="linear_regression_v_test",
        name="LinearRegression_OLS",
        version="test",
        kind=ModelKind.LIBRARY,
        hyperparams=lin_hp,
    ))

    # Register metrics
    zoo.register_metric(EvalMetricSpec(
        metric_id="r2_v_test", name="R2_Score", version="test",
        direction=MetricDirection.HIGHER_IS_BETTER, kind=MetricKind.LIBRARY))
    zoo.register_metric(EvalMetricSpec(
        metric_id="mse_v_test", name="MeanSquaredError", version="test",
        direction=MetricDirection.LOWER_IS_BETTER, kind=MetricKind.LIBRARY))

    # Bindings
    lin_binding = ExecutionBinding(
        bound_id="linear_regression_v_test", kind=BindingKind.LIBRARY,
        module="model_zoo.examples.models", callable="linreg_train_and_infer")
    r2_binding = ExecutionBinding(
        bound_id="r2_v_test", kind=BindingKind.LIBRARY,
        module="model_zoo.examples.metrics", callable="r2_score")
    mse_binding = ExecutionBinding(
        bound_id="mse_v_test", kind=BindingKind.LIBRARY,
        module="model_zoo.examples.metrics", callable="mse")

    # Inputs
    X = [i / 10 for i in range(1, 51)]
    y = [1.0 + 2.0 * x for x in X]

    # Submit
    req = ModelRunRequest(
        model_id="linear_regression_v_test",
        inputs={"X": DataRef(DataRefType.INLINE, inline=X),
                "y": DataRef(DataRefType.INLINE, inline=y)},
        params={"fit_intercept": True, "feature_name": "X"},
        binding=lin_binding,
        evaluation=EvaluationSpec(
            metric_ids=["r2_v_test", "mse_v_test"],
            y_true_key="y",
            train_indices=list(range(40)), test_indices=list(range(40, 50)),
            bindings={"r2_v_test": r2_binding, "mse_v_test": mse_binding},
        ),
    )
    h = zoo.submit_run(req)
    # Poll until finished (fast path)
    import time
    for _ in range(100):
        res = zoo.get_run_result(h.run_id)
        if res.status.name != "STARTED" and not res.metadata.get("in_progress"):
            break
        time.sleep(0.01)

    assert res.status.name == "SUCCEEDED"
    assert res.summary
    assert "Coefficients" in res.summary
    assert "Metrics" in res.summary and "train" in res.summary["Metrics"]
