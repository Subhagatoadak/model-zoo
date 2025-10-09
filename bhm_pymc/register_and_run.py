from model_zoo import (
    ModelZoo, ModelRegistry, MetricRegistry,
    ModelSpec, ModelKind, EvalMetricSpec, MetricKind, MetricDirection,
    HyperParamSpace, HyperParamField, DType,
    ExecutionBinding, BindingKind,
    DataRef, DataRefType, ExecutionContext,
    ModelRunRequest, MetricComputeRequest, EvaluationSpec,
    InMemoryBindingRegistry, InMemoryRunStore,
)
from model_zoo.runners.library import LibraryModelRunner, LibraryMetricRunner
from model_zoo.examples.models import make_synthetic_linear, make_synthetic_mtcars_wt_mpg, make_synthetic_mtcars_multi
from model_zoo import r_style_summary_string
from model_zoo.utils.logging import configure_logging, get_logger, quiet_third_party_loggers
from model_zoo.examples.sklearn_models import sk_lr_train_and_infer  # noqa: F401 (for binding importability)
from model_zoo.examples.pymc_models import pymc_linear_nuts  # noqa: F401 (for binding importability)
from model_zoo.examples import metrics as example_metrics
from model_zoo.contracts.validation import ParamValidationService
from model_zoo import DefaultParamValidator
from model_zoo.contracts.audit import AuditLog, AuditEvent

class PrintAudit(AuditLog):
    def append(self, event: AuditEvent) -> None:
        print(f"[AUDIT] {event.kind} ref={event.ref_id}")

def main():
    configure_logging("INFO")
    quiet_third_party_loggers()
    log = get_logger(__name__)
    log.info("Starting example run")
    model_reg, metric_reg = ModelRegistry(), MetricRegistry()
    # Configure runner with an in-memory store; attach binding registry to Zoo
    model_runner = LibraryModelRunner()
    model_runner.set_run_store(InMemoryRunStore())
    bind_reg = InMemoryBindingRegistry()
    zoo = ModelZoo(model_reg, metric_reg,
                   binding_registry=bind_reg,
                   model_runner=model_runner,
                   metric_runner=LibraryMetricRunner(),
                   validator=DefaultParamValidator(),
                   audit=PrintAudit())
    

    # Define two library models: OLS linear regression and Bayesian linear regression
    lin_hp = HyperParamSpace(fields={
        "fit_intercept": HyperParamField("fit_intercept", DType.BOOL, default=True, required=False,
                                          description="Include intercept in linear model"),
    })

    bayes_hp = HyperParamSpace(fields={
        "sigma": HyperParamField("sigma", DType.FLOAT, default=0.5, required=True,
                                  description="Observation noise std"),
        "prior_mu": HyperParamField("prior_mu", DType.FLOAT, default=0.0, required=False,
                                     description="Prior mean for slope"),
        "prior_tau": HyperParamField("prior_tau", DType.FLOAT, default=10.0, required=False,
                                      description="Prior std for slope"),
        "fit_intercept": HyperParamField("fit_intercept", DType.BOOL, default=True, required=False),
        "draws": HyperParamField("draws", DType.INT, default=0, required=False,
                                  description="Number of posterior samples to draw for diagnostics"),
        "tune": HyperParamField("tune", DType.INT, default=0, required=False,
                                 description="Tuning steps (ignored in conjugate demo)"),
        "priors": HyperParamField(
            "priors", DType.DICT, default=None, required=False,
            description=(
                "Structured priors override. Example: "
                "{'beta': {'distribution': 'normal', 'mu': 0.0, 'tau': 5.0}, "
                " 'sigma': {'distribution': 'gamma', 'shape': 2.0, 'rate': 1.0}}"
            ),
            example={
                "beta": {"distribution": "normal", "mu": 0.0, "tau": 5.0},
                "sigma": {"distribution": "gamma", "shape": 2.0, "rate": 1.0},
            },
        ),
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

    # Register model and metric bindings in registry (so requests needn't carry them)
    bind_reg.register_model_binding("linear_regression_v1", lin_binding)
    bind_reg.register_model_binding("bayes_linear_v1", bayes_binding)
    bind_reg.register_metric_binding("r2_v1", r2_binding)
    bind_reg.register_metric_binding("mse_v1", mse_binding)

    # Run linear regression with training on train_idx and evaluate on test_idx
    lin_req = ModelRunRequest(
        model_id="linear_regression_v1",
        inputs={"X": X_ref, "y": y_ref},
        params={"fit_intercept": True, "train_indices": train_idx, "feature_name": "X"},
        context=ExecutionContext(random_seed=2025),
        dataset_version="synthetic:v1",
        evaluation=EvaluationSpec(
            metric_ids=["r2_v1", "mse_v1"],
            y_true_key="y",
            compute_train=True,
            compute_valid=True,
            train_indices=train_idx,
            test_indices=test_idx,
            bindings={
                "r2_v1": r2_binding,
                "mse_v1": mse_binding,
            }
        ),
    )
    def wait_for_result(zoo_obj, run_id: str, actor: str = "you", timeout_s: float = 300.0):
        import time
        start = time.monotonic()
        sleep = 0.05
        last = None
        while True:
            res = zoo_obj.get_run_result(run_id, actor=actor)
            last = res
            # Break when terminal status or in_progress flag cleared
            if res.status.name in ("SUCCEEDED", "FAILED", "CANCELLED"):
                return res
            if not res.metadata.get("in_progress"):
                return res
            if time.monotonic() - start > timeout_s:
                return res
            time.sleep(sleep)
            sleep = min(sleep * 1.5, 1.0)
        return last

    try:
        lin_handle = zoo.submit_run(lin_req, actor="you")
        lin_res = wait_for_result(zoo, lin_handle.run_id, actor="you")
    except Exception:
        log.exception("Linear run failed")
        raise
    print("LinearRegression: status=", lin_res.status.value)
    print(r_style_summary_string(lin_res.summary))

    # Switch to Bayesian linear regression (swap model_id + binding + params) with same split
    bayes_req = ModelRunRequest(
        model_id="bayes_linear_v1",
        inputs={"X": X_ref, "y": y_ref},
        params={
            "sigma": 0.5,
            "prior_mu": 0.0,
            "prior_tau": 5.0,
            "fit_intercept": True,
            "train_indices": train_idx,
            "draws": 200,
            "tune": 100,
            "priors": {"beta": {"mu": 0.0, "tau": 5.0}},
            "feature_name": "X",
        },
        context=ExecutionContext(random_seed=2025),
        dataset_version="synthetic:v1",
        evaluation=EvaluationSpec(
            metric_ids=["r2_v1", "mse_v1"],
            y_true_key="y",
            compute_train=True,
            compute_valid=True,
            train_indices=train_idx,
            test_indices=test_idx,
            bindings={
                "r2_v1": r2_binding,
                "mse_v1": mse_binding,
            }
        ),
    )
    try:
        bayes_handle = zoo.submit_run(bayes_req, actor="you")
        bayes_res = wait_for_result(zoo, bayes_handle.run_id, actor="you")
    except Exception:
        log.exception("Bayesian run failed")
        raise
    print("BayesLinear: status=", bayes_res.status.value)
    print(r_style_summary_string(bayes_res.summary))

    # Run Bayesian model again showing explicit priors configuration in hyperparameters
    # This time include a Gamma prior on sigma (precision conjugacy via Normal-Gamma)
    custom_priors = {
        "beta": {"distribution": "normal", "mu": 0.5, "tau": 2.0},
        "sigma": {"distribution": "gamma", "shape": 3.0, "rate": 2.0},
    }
    bayes_req_custom = ModelRunRequest(
        model_id="bayes_linear_v1",
        inputs={"X": X_ref, "y": y_ref},
        params={
            "sigma": 0.5,
            "prior_mu": 0.0,      # can be omitted when using 'priors'
            "prior_tau": 10.0,    # can be omitted when using 'priors'
            "fit_intercept": True,
            "train_indices": train_idx,
            "draws": 100,
            "priors": custom_priors,
            "feature_name": "X",
        },
        context=ExecutionContext(random_seed=2025),
        dataset_version="synthetic:v1",
        evaluation=EvaluationSpec(
            metric_ids=["r2_v1", "mse_v1"],
            y_true_key="y",
            compute_train=True,
            compute_valid=True,
            train_indices=train_idx,
            test_indices=test_idx,
            bindings={
                "r2_v1": r2_binding,
                "mse_v1": mse_binding,
            }
        ),
    )
    try:
        bayes_handle2 = zoo.submit_run(bayes_req_custom, actor="you")
        bayes_res2 = wait_for_result(zoo, bayes_handle2.run_id, actor="you")
    except Exception:
        log.exception("Bayesian run (custom priors) failed")
        raise
    print("BayesLinear (custom priors): status=", bayes_res2.status.value)
    print(r_style_summary_string(bayes_res2.summary))
    
    
    # ------------------------------------------------------------------
    # Bayesian model with synthetic mtcars-like data (mpg ~ wt) and custom priors
    wt, mpg = make_synthetic_mtcars_wt_mpg(n=32, alpha=36.0, beta=-4.5, noise_sigma=2.0)
    wt_ref = DataRef(ref_type=DataRefType.INLINE, inline=wt)
    mpg_ref = DataRef(ref_type=DataRefType.INLINE, inline=mpg)
    n_m = len(wt)
    split_m = int(0.75 * n_m)
    tr_idx_m = list(range(split_m))
    te_idx_m = list(range(split_m, n_m))

    mtcars_spec = ModelSpec(
        model_id="bayes_mtcars_wt_v1",
        name="BayesianLinearRegression_mtcars_wt",
        version="1.0.0",
        kind=ModelKind.LIBRARY,
        hyperparams=bayes_hp,
        tags=["bayesian", "mtcars", "regression"],
        metadata={"example": True},
    )
    zoo.register_model(mtcars_spec, actor="you")

    mtcars_binding = ExecutionBinding(
        bound_id="bayes_mtcars_wt_v1", kind=BindingKind.LIBRARY,
        module="model_zoo.examples.models", callable="bayes_linreg_train_and_infer",
    )
    bind_reg.register_model_binding("bayes_mtcars_wt_v1", mtcars_binding)
    mtcars_req = ModelRunRequest(
        model_id="bayes_mtcars_wt_v1",
        inputs={"X": wt_ref, "y": mpg_ref},
        params={
            "sigma": 2.0,
            "fit_intercept": True,
            "train_indices": tr_idx_m,
            "draws": 200,
            "priors": {
                "beta": {"distribution": "normal", "mu": -4.0, "tau": 3.0},
                "sigma": {"distribution": "gamma", "shape": 3.0, "rate": 2.0},
            },
            "feature_name": "wt",
        },
        context=ExecutionContext(random_seed=2025),
        dataset_version="synthetic_mtcars:v1",
        evaluation=EvaluationSpec(
            metric_ids=["r2_v1", "mse_v1"],
            y_true_key="y",
            compute_train=True,
            compute_valid=True,
            train_indices=tr_idx_m,
            test_indices=te_idx_m,
            bindings={"r2_v1": r2_binding, "mse_v1": mse_binding},
        ),
    )
    try:
        mt_handle = zoo.submit_run(mtcars_req, actor="you")
        mt_res = wait_for_result(zoo, mt_handle.run_id, actor="you")
        print("Bayes (mtcars mpg~wt): status=", mt_res.status.value)
        print(r_style_summary_string(mt_res.summary))
    except Exception as e:
        log.exception("Bayesian mtcars example failed: %s", e)

    # ------------------------------------------------------------------
    # Multivariate mtcars-like: mpg ~ wt + hp + disp
    X_rows, y_m, feat_names = make_synthetic_mtcars_multi()
    X_rows_ref = DataRef(ref_type=DataRefType.INLINE, inline=X_rows)
    y_m_ref = DataRef(ref_type=DataRefType.INLINE, inline=y_m)
    n_mv = len(X_rows)
    split_mv = int(0.75 * n_mv)
    tr_idx_mv = list(range(split_mv))
    te_idx_mv = list(range(split_mv, n_mv))

    # Sklearn multivariate
    sk_multi_spec = ModelSpec(
        model_id="sklearn_linear_mtcars_v1",
        name="SklearnLinearRegression_MTCarsMulti",
        version="1.0.0",
        kind=ModelKind.LIBRARY,
        hyperparams=HyperParamSpace(fields={
            "fit_intercept": HyperParamField("fit_intercept", DType.BOOL, default=True),
            "feature_names": HyperParamField("feature_names", DType.LIST, default=None),
        }),
        tags=["sklearn", "linear", "mtcars"],
        metadata={"example": True}
    )
    zoo.register_model(sk_multi_spec, actor="you")
    sk_multi_binding = ExecutionBinding(
        bound_id="sklearn_linear_mtcars_v1", kind=BindingKind.LIBRARY,
        module="model_zoo.examples.sklearn_models", callable="sk_lr_train_and_infer",
    )
    sk_multi_req = ModelRunRequest(
        model_id="sklearn_linear_mtcars_v1",
        inputs={"X": X_rows_ref, "y": y_m_ref},
        params={"fit_intercept": True, "train_indices": tr_idx_mv, "feature_names": feat_names},
        binding=sk_multi_binding,
        context=ExecutionContext(random_seed=2025),
        evaluation=EvaluationSpec(
            metric_ids=["r2_v1", "mse_v1"],
            y_true_key="y",
            compute_train=True,
            compute_valid=True,
            train_indices=tr_idx_mv,
            test_indices=te_idx_mv,
            bindings={"r2_v1": r2_binding, "mse_v1": mse_binding},
        ),
    )
    try:
        skm_handle = zoo.submit_run(sk_multi_req, actor="you")
        skm_res = wait_for_result(zoo, skm_handle.run_id, actor="you")
        print("SklearnLinear (mtcars multi): status=", skm_res.status.value)
        print(r_style_summary_string(skm_res.summary))
    except Exception as e:
        log.warning("Sklearn multivariate example skipped: %s", e)

    # PyMC multivariate NUTS
    pymc_multi_spec = ModelSpec(
        model_id="pymc_linear_nuts_mtcars_v1",
        name="PyMCLinearNUTS_MTCarsMulti",
        version="1.0.0",
        kind=ModelKind.LIBRARY,
        hyperparams=HyperParamSpace(fields={
            "draws": HyperParamField("draws", DType.INT, default=500),
            "tune": HyperParamField("tune", DType.INT, default=500),
            "target_accept": HyperParamField("target_accept", DType.FLOAT, default=0.9),
            "priors": HyperParamField("priors", DType.DICT, default=None),
            "feature_names": HyperParamField("feature_names", DType.LIST, default=None),
        }),
        tags=["pymc", "nuts", "bayesian", "mtcars"],
        metadata={"example": True}
    )
    zoo.register_model(pymc_multi_spec, actor="you")
    pymc_multi_binding = ExecutionBinding(
        bound_id="pymc_linear_nuts_mtcars_v1", kind=BindingKind.LIBRARY,
        module="model_zoo.examples.pymc_models", callable="pymc_linear_nuts",
    )
    bind_reg.register_model_binding("pymc_linear_nuts_mtcars_v1", pymc_multi_binding)
    # Custom per-feature priors for PyMC multivariate (match feature_names order)
    beta_mu_map = {"wt": -4.0, "hp": -0.01, "disp": -0.005}
    beta_tau_map = {"wt": 3.0, "hp": 0.05, "disp": 0.02}
    beta_mu_list = [beta_mu_map.get(n, 0.0) for n in feat_names]
    beta_tau_list = [beta_tau_map.get(n, 10.0) for n in feat_names]

    pymc_multi_req = ModelRunRequest(
        model_id="pymc_linear_nuts_mtcars_v1",
        inputs={"X": X_rows_ref, "y": y_m_ref},
        params={
            "draws": 300,
            "tune": 300,
            "target_accept": 0.9,
            "chains": 2,
            "cores": 1,
            "compute_convergence_checks": True,
            "train_indices": tr_idx_mv,
            "feature_names": feat_names,
            # Custom per-feature priors for beta (Normal) and sigma (Gamma/Erlang)
            "priors": {
                "beta": {"distribution": "normal", "mu": beta_mu_list, "tau": beta_tau_list},
                "sigma": {"distribution": "erlang", "shape": 3.0, "rate": 2.0},
            },
        },
        context=ExecutionContext(random_seed=2025),
        evaluation=EvaluationSpec(
            metric_ids=["r2_v1", "mse_v1"],
            y_true_key="y",
            compute_train=True,
            compute_valid=True,
            train_indices=tr_idx_mv,
            test_indices=te_idx_mv,
            bindings={"r2_v1": r2_binding, "mse_v1": mse_binding},
        ),
    )
    try:
        pmm_handle = zoo.submit_run(pymc_multi_req, actor="you")
        pmm_res = wait_for_result(zoo, pmm_handle.run_id, actor="you")
        print("PyMC NUTS (mtcars multi): status=", pmm_res.status.value)
        print(r_style_summary_string(pmm_res.summary))
    except Exception as e:
        log.warning("PyMC multivariate example skipped: %s", e)

    # scikit-learn LinearRegression example (if sklearn installed)
    sk_spec = ModelSpec(
        model_id="sklearn_linear_v1",
        name="SklearnLinearRegression",
        version="1.0.0",
        kind=ModelKind.LIBRARY,
        hyperparams=HyperParamSpace(fields={
            "fit_intercept": HyperParamField("fit_intercept", DType.BOOL, default=True),
        }),
        tags=["sklearn", "linear"],
        metadata={"example": True}
    )
    zoo.register_model(sk_spec, actor="you")
    sk_binding = ExecutionBinding(
        bound_id="sklearn_linear_v1", kind=BindingKind.LIBRARY,
        module="model_zoo.examples.sklearn_models", callable="sk_lr_train_and_infer",
    )
    bind_reg.register_model_binding("sklearn_linear_v1", sk_binding)
    sk_req = ModelRunRequest(
        model_id="sklearn_linear_v1",
        inputs={"X": X_ref, "y": y_ref},
        params={"fit_intercept": True, "train_indices": train_idx, "feature_names": ["X"]},
        context=ExecutionContext(random_seed=2025),
        evaluation=EvaluationSpec(
            metric_ids=["r2_v1", "mse_v1"],
            y_true_key="y",
            compute_train=True,
            compute_valid=True,
            train_indices=train_idx,
            test_indices=test_idx,
            bindings={"r2_v1": r2_binding, "mse_v1": mse_binding},
        ),
    )
    try:
        sk_handle = zoo.submit_run(sk_req, actor="you")
        sk_res = wait_for_result(zoo, sk_handle.run_id, actor="you")
        print("SklearnLinear: status=", sk_res.status.value)
        print(r_style_summary_string(sk_res.summary))
    except Exception as e:
        log.warning("Sklearn example skipped: %s", e)

   

if __name__ == "__main__":
    main()
