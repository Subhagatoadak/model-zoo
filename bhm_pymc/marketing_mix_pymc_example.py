"""
A clean, self‑contained example demonstrating a Bayesian marketing mix model
using the Model Zoo facade and a PyMC NUTS backend.

What this shows
- Synthetic marketing mix data generation (sales ~ TV + Radio + Social spend)
- Registering a library model that uses PyMC for Bayesian linear regression
- Custom priors per feature (beta ~ Normal; sigma ~ HalfNormal/Gamma)
- In‑process Library runners, BindingRegistry, in‑memory RunStore
- Submitting a run with an EvaluationSpec (train/test split), computing R2/MSE
- Printing an R‑style summary with named coefficients

Note: This example requires `pymc` and `numpy`. If those are not available, the
example will skip model execution gracefully with a helpful message.
"""
from __future__ import annotations

# 1) Imports: Model Zoo facade and building blocks
from model_zoo import (
    ModelZoo, ModelRegistry, MetricRegistry,
    ModelSpec, ModelKind, EvalMetricSpec, MetricKind, MetricDirection,
    HyperParamSpace, HyperParamField, DType,
    ExecutionBinding, BindingKind,
    DataRef, DataRefType, ExecutionContext,
    ModelRunRequest, EvaluationSpec,
    InMemoryBindingRegistry, InMemoryRunStore,
    r_style_summary_string,
)
from model_zoo.runners.library import LibraryModelRunner, LibraryMetricRunner
from model_zoo import DefaultParamValidator
from model_zoo.utils.logging import configure_logging, quiet_third_party_loggers, get_logger

# 2) Import the PyMC runner callable and example metrics
from model_zoo.examples.pymc_models import pymc_linear_nuts  # noqa: F401 (binding target)
from model_zoo.examples import metrics as example_metrics  # noqa: F401 (binding targets)


# 3) Synthetic data: simple linear marketing mix
def make_synthetic_marketing_mix(n: int = 52, seed: int = 42):
    """Create weekly synthetic marketing data for TV/Radio/Social and Sales.

    Sales = alpha + b_tv * TV + b_radio * Radio + b_social * Social + noise
    Coefficients encode channel efficiency; set up with modest noise.
    """
    import random
    random.seed(seed)

    # Weekly spends in arbitrary units
    tv = [random.uniform(5.0, 30.0) for _ in range(n)]
    radio = [random.uniform(1.0, 10.0) for _ in range(n)]
    social = [random.uniform(0.5, 8.0) for _ in range(n)]

    # Ground truth coefficients
    alpha = 100.0
    b_tv, b_radio, b_social = 2.0, 1.0, 0.5
    noise_sigma = 5.0

    sales = [
        alpha + b_tv * tv[i] + b_radio * radio[i] + b_social * social[i] + random.gauss(0, noise_sigma)
        for i in range(n)
    ]

    # X as list of dicts -> keeps names alongside values; runner will derive feature_names
    X_rows = [{"tv": tv[i], "radio": radio[i], "social": social[i]} for i in range(n)]
    return X_rows, sales, ["tv", "radio", "social"]


def main():
    # 4) Set up logging (INFO) and quiet noisy third‑party logs
    configure_logging("INFO")
    quiet_third_party_loggers()
    log = get_logger(__name__)
    log.info("Starting Bayesian marketing mix example")

    # 5) Core Model Zoo wiring: registries, runner with a run store, binding registry, validator
    model_reg, metric_reg = ModelRegistry(), MetricRegistry()
    model_runner = LibraryModelRunner()
    model_runner.set_run_store(InMemoryRunStore())  # persist run results in memory for this process
    metric_runner = LibraryMetricRunner()
    bind_reg = InMemoryBindingRegistry()

    zoo = ModelZoo(
        model_registry=model_reg,
        metric_registry=metric_reg,
        model_runner=model_runner,
        metric_runner=metric_runner,
        validator=DefaultParamValidator(),
        binding_registry=bind_reg,
    )

    # 6) Register a PyMC Bayesian linear regression model spec
    # Explain: the registry carries only metadata and a hyperparameter schema; it does not hold code.
    pymc_hp = HyperParamSpace(fields={
        "draws": HyperParamField("draws", DType.INT, default=500, description="Posterior draws"),
        "tune": HyperParamField("tune", DType.INT, default=500, description="Warmup iterations"),
        "target_accept": HyperParamField("target_accept", DType.FLOAT, default=0.9),
        "feature_names": HyperParamField("feature_names", DType.LIST, default=None),
        # Prior configuration objects (kept flexible for agent control)
        "priors": HyperParamField("priors", DType.DICT, default=None, description="beta/sigma priors"),
    })
    model_spec = ModelSpec(
        model_id="pymc_marketing_mix_v1",
        name="BayesianLinear_PyMC_MarketingMix",
        version="1.0.0",
        kind=ModelKind.LIBRARY,
        hyperparams=pymc_hp,
        tags=["bayesian", "pymc", "marketing"],
        metadata={"example": True},
    )
    zoo.register_model(model_spec, actor="example")

    # 7) Register evaluation metrics: R2 (higher is better) and MSE (lower is better)
    r2_spec = EvalMetricSpec("r2_v1", "R2_Score", "1.0.0", MetricDirection.HIGHER_IS_BETTER, MetricKind.LIBRARY)
    mse_spec = EvalMetricSpec("mse_v1", "MeanSquaredError", "1.0.0", MetricDirection.LOWER_IS_BETTER, MetricKind.LIBRARY)
    zoo.register_metric(r2_spec, actor="example")
    zoo.register_metric(mse_spec, actor="example")

    # 8) Bindings: how to execute the model and metrics (decoupled from registry)
    # Explain: Binding registry resolves the concrete execution at submit time; specs stay pure.
    model_binding = ExecutionBinding(
        bound_id="pymc_marketing_mix_v1",
        kind=BindingKind.LIBRARY,
        module="model_zoo.examples.pymc_models",
        callable="pymc_linear_nuts",
    )
    r2_binding = ExecutionBinding("r2_v1", BindingKind.LIBRARY, module="model_zoo.examples.metrics", callable="r2_score")
    mse_binding = ExecutionBinding("mse_v1", BindingKind.LIBRARY, module="model_zoo.examples.metrics", callable="mse")
    bind_reg.register_model_binding("pymc_marketing_mix_v1", model_binding)
    bind_reg.register_metric_binding("r2_v1", r2_binding)
    bind_reg.register_metric_binding("mse_v1", mse_binding)

    # 9) Generate synthetic marketing mix data and build DataRefs
    X_rows, sales, feat_names = make_synthetic_marketing_mix()
    X_ref = DataRef(DataRefType.INLINE, inline=X_rows)
    y_ref = DataRef(DataRefType.INLINE, inline=sales)

    # Train/test split (e.g., 75/25)
    n = len(sales)
    split = int(0.75 * n)
    train_idx = list(range(split))
    test_idx = list(range(split, n))

    # 10) Custom priors per feature
    # Why: encodes domain expectations (e.g., positive TV elasticity) and stabilizes inference
    # Example: different distributions per channel
    # - tv: Normal prior around 2.0 (moderate effect), tau=1.0
    # - radio: HalfNormal (non-negative effect), scale=1.0
    # - social: Gamma/Erlang (strictly positive), shape=2.0, rate=1.0
    priors = {
        "beta": {
            "distribution": ["normal", "half_normal", "gamma"],
            "mu": [2.0, None, None],           # used only for normal
            "tau": [1.0, None, None],          # used only for normal
            "scale": [None, 1.0, None],        # used only for half_normal
            "shape": [None, None, 2.0],        # used only for gamma/erlang
            "rate": [None, None, 1.0],
        },
        # Sigma prior: choose HalfNormal (scale) or Gamma/Erlang (shape, rate)
        "sigma": {"distribution": "half_normal", "scale": 5.0},
    }

    # 11) Build a run request with EvaluationSpec
    # Explain: EvaluationSpec asks the platform to compute metrics inline and return them in the summary
    req = ModelRunRequest(
        model_id="pymc_marketing_mix_v1",
        inputs={"X": X_ref, "y": y_ref},
        params={
            "draws": 400,                # posterior samples to draw
            "tune": 400,                 # warmup for NUTS
            "target_accept": 0.9,        # robust proposals
            "chains": 2, "cores": 1,    # deterministic(ish) in simple environments
            "compute_convergence_checks": True,
            "feature_names": feat_names,
            "priors": priors,
        },
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

    # 12) Submit the run; the Zoo will resolve the binding from the registry and execute via the Library runner
    def wait_for_result(zoo_obj, run_id: str, actor: str = "example", timeout_s: float = 600.0):
        import time
        start, delay = time.monotonic(), 0.05
        while True:
            res = zoo_obj.get_run_result(run_id, actor=actor)
            if res.status.name in ("SUCCEEDED", "FAILED", "CANCELLED"):
                return res
            if not res.metadata.get("in_progress"):
                return res
            if time.monotonic() - start > timeout_s:
                return res
            time.sleep(delay)
            delay = min(1.0, delay * 1.5)

    try:
        handle = zoo.submit_run(req, actor="example")
        res = wait_for_result(zoo, handle.run_id)
    except Exception as e:
        # If pymc/numpy wheels are not present, we skip gracefully
        log.warning("PyMC marketing mix example skipped: %s", e)
        return

    # 13) Print results in an R‑style summary for fast inspection
    print("PyMC Marketing Mix: status=", res.status.value)
    print(r_style_summary_string(res.summary))


if __name__ == "__main__":
    main()
