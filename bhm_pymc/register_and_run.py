from model_zoo import (
    ModelZoo, ModelRegistry, MetricRegistry,
    ModelSpec, ModelKind, EvalMetricSpec, MetricKind, MetricDirection,
    HyperParamSpace, HyperParamField, DType,
    ExecutionBinding, BindingKind,
    DataRef, DataRefType, ExecutionContext,
    ModelRunRequest, MetricComputeRequest, EvaluationSpec,
)
from model_zoo.runners.library import LibraryModelRunner, LibraryMetricRunner
from model_zoo.examples.models import make_synthetic_linear
from model_zoo.examples import metrics as example_metrics
from model_zoo.contracts.validation import ParamValidationService
from model_zoo.contracts.audit import AuditLog, AuditEvent

class NoOpValidator(ParamValidationService):
    def validate_model_params(self, model_spec, params): return None
    def validate_metric_params(self, metric_spec, params): return None

class PrintAudit(AuditLog):
    def append(self, event: AuditEvent) -> None:
        print(f"[AUDIT] {event.kind} ref={event.ref_id}")

def main():
    model_reg, metric_reg = ModelRegistry(), MetricRegistry()
    zoo = ModelZoo(model_reg, metric_reg,
                   binding_registry=None,  # not needed; we pass binding on request in this demo
                   model_runner=LibraryModelRunner(),
                   metric_runner=LibraryMetricRunner(),
                   validator=NoOpValidator(),
                   audit=PrintAudit())
    

    # Define two library models: OLS linear regression and Bayesian linear regression
    lin_hp = HyperParamSpace(fields={
        "fit_intercept": HyperParamField("fit_intercept", DType.BOOL, default=True, required=False,
                                          description="Include intercept in linear model"),
    })

    bayes_hp = HyperParamSpace(fields={
        "sigma": HyperParamField("sigma", DType.FLOAT, default=0.5, required=True,
                                  description="Observation noise std"),
        "prior_tau": HyperParamField("prior_tau", DType.FLOAT, default=10.0, required=False,
                                      description="Prior std for slope"),
        "fit_intercept": HyperParamField("fit_intercept", DType.BOOL, default=True, required=False),
    })

    lin_spec = ModelSpec(
        model_id="linear_regression_v1",
        name="LinearRegression_OLS",
        version="1.0.0",
        kind=ModelKind.LIBRARY,
        hyperparams=lin_hp,
        tags=["linear", "ols"],
        metadata={"example": True}
    )

    bayes_spec = ModelSpec(
        model_id="bayes_linear_v1",
        name="BayesianLinearRegression_Conjugate",
        version="1.0.0",
        kind=ModelKind.LIBRARY,
        hyperparams=bayes_hp,
        tags=["bayesian", "regression"],
        metadata={"example": True}
    )

    zoo.register_model(lin_spec, actor="you")
    zoo.register_model(bayes_spec, actor="you")

    # Register evaluation metrics (separate from model outputs)
    r2_spec = EvalMetricSpec(
        metric_id="r2_v1",
        name="R2_Score",
        version="1.0.0",
        direction=MetricDirection.HIGHER_IS_BETTER,
        kind=MetricKind.LIBRARY,
        tags=["regression", "goodness_of_fit"],
    )
    mse_spec = EvalMetricSpec(
        metric_id="mse_v1",
        name="MeanSquaredError",
        version="1.0.0",
        direction=MetricDirection.LOWER_IS_BETTER,
        kind=MetricKind.LIBRARY,
        tags=["regression", "error"],
    )
    zoo.register_metric(r2_spec, actor="you")
    zoo.register_metric(mse_spec, actor="you")

    # List models
    print("Registered models:")
    for m in zoo.list_models():
        print(" -", m.model_id, m.name, m.version)
    print("Registered metrics:")
    for mt in zoo.list_metrics():
        print(" -", mt.metric_id, mt.name, mt.version)

    # Prepare synthetic dataset (INLINE DataRefs)
    X, y = make_synthetic_linear(n=100, slope=2.0, intercept=1.0, noise_sigma=0.5)
    X_ref = DataRef(ref_type=DataRefType.INLINE, inline=X)
    y_ref = DataRef(ref_type=DataRefType.INLINE, inline=y)
    # Train/test split indices (80/20)
    n = len(X)
    split = int(0.8 * n)
    train_idx = list(range(0, split))
    test_idx = list(range(split, n))

    # Bindings to example in-process callables
    lin_binding = ExecutionBinding(
        bound_id="linear_regression_v1", kind=BindingKind.LIBRARY,
        module="model_zoo.examples.models", callable="linreg_train_and_infer",
    )
    bayes_binding = ExecutionBinding(
        bound_id="bayes_linear_v1", kind=BindingKind.LIBRARY,
        module="model_zoo.examples.models", callable="bayes_linreg_train_and_infer",
    )

    # Metric bindings
    r2_binding = ExecutionBinding(
        bound_id="r2_v1", kind=BindingKind.LIBRARY,
        module="model_zoo.examples.metrics", callable="r2_score",
    )
    mse_binding = ExecutionBinding(
        bound_id="mse_v1", kind=BindingKind.LIBRARY,
        module="model_zoo.examples.metrics", callable="mse",
    )

    # Run linear regression with training on train_idx and evaluate on test_idx
    lin_req = ModelRunRequest(
        model_id="linear_regression_v1",
        inputs={"X": X_ref, "y": y_ref},
        params={"fit_intercept": True, "train_indices": train_idx},
        context=ExecutionContext(random_seed=2025),
        dataset_version="synthetic:v1",
        binding=lin_binding,
        evaluation=EvaluationSpec(
            metric_ids=["r2_v1", "mse_v1"],
            y_true_key="y",
            test_indices=test_idx,
            bindings={
                "r2_v1": r2_binding,
                "mse_v1": mse_binding,
            }
        ),
    )
    def wait_for_result(zoo_obj, run_id: str, actor: str = "you", max_tries: int = 100):
        import time
        res = None
        for _ in range(max_tries):
            res = zoo_obj.get_run_result(run_id, actor=actor)
            if not res.metadata.get("in_progress"):
                return res
            time.sleep(0.01)
        return res

    lin_handle = zoo.submit_run(lin_req, actor="you")
    lin_res = wait_for_result(zoo, lin_handle.run_id, actor="you")
    print("LinearRegression: status=", lin_res.status.value, "coefficients=", lin_res.diagnostics)
    # Metrics computed automatically based on evaluation spec
    print("LinearRegression metrics:", lin_res.eval_scores)

    # Switch to Bayesian linear regression (swap model_id + binding + params) with same split
    bayes_req = ModelRunRequest(
        model_id="bayes_linear_v1",
        inputs={"X": X_ref, "y": y_ref},
        params={"sigma": 0.5, "prior_tau": 5.0, "fit_intercept": True, "train_indices": train_idx},
        context=ExecutionContext(random_seed=2025),
        dataset_version="synthetic:v1",
        binding=bayes_binding,
        evaluation=EvaluationSpec(
            metric_ids=["r2_v1", "mse_v1"],
            y_true_key="y",
            test_indices=test_idx,
            bindings={
                "r2_v1": r2_binding,
                "mse_v1": mse_binding,
            }
        ),
    )
    bayes_handle = zoo.submit_run(bayes_req, actor="you")
    bayes_res = wait_for_result(zoo, bayes_handle.run_id, actor="you")
    print("BayesLinear: status=", bayes_res.status.value, "params=", bayes_res.diagnostics)
    print("BayesLinear metrics:", bayes_res.eval_scores)

if __name__ == "__main__":
    main()
