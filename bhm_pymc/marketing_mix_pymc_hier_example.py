"""
Hierarchical Bayesian Marketing Mix Example (PyMC NUTS)

This example demonstrates a partial-pooling marketing mix model with varying
intercepts by geography and brand. It uses the Model Zoo facade with a
BindingRegistry and in-memory RunStore, and runs an in-process PyMC model.

Steps explained in detail with comments:
1) Create synthetic weekly data: features tv/radio/social, categories geo/brand
2) Wire up Model Zoo (registries, runner, binding registry, validator)
3) Register a hierarchical PyMC model spec and metric specs
4) Register bindings and resolve them via BindingRegistry at submit time
5) Build a ModelRunRequest with custom per-feature priors and EvaluationSpec
6) Submit, wait, and print an R-style summary with named coefficients

Requirements: pymc, numpy. If missing, execution is skipped gracefully.
"""
from __future__ import annotations

from model_zoo import (
    ModelZoo, ModelRegistry, MetricRegistry,
    ModelSpec, ModelKind, EvalMetricSpec, MetricKind, MetricDirection,
    HyperParamSpace, HyperParamField, DType,
    ExecutionBinding, BindingKind,
    DataRef, DataRefType, ExecutionContext,
    ModelRunRequest, EvaluationSpec,
    InMemoryBindingRegistry, InMemoryRunStore, InMemoryEventBus,
    r_style_summary_string,
)
from model_zoo.runners.library import LibraryModelRunner, LibraryMetricRunner
from model_zoo import DefaultParamValidator
from model_zoo.utils.logging import configure_logging, quiet_third_party_loggers, get_logger

# Ensure binding targets are importable
from model_zoo.examples.pymc_models import pymc_hierarchical_mmm  # noqa: F401
from model_zoo.examples import metrics as example_metrics  # noqa: F401


def make_synthetic_marketing_hier(n_weeks: int = 52, n_geo: int = 5, n_brand: int = 3, seed: int = 7):
    """Generate synthetic MMM data with geography and brand random intercepts.

    Sales = alpha + a_geo[geo] + a_brand[brand] + b_tv*TV + b_radio*Radio + b_social*Social + eps
    """
    import random
    random.seed(seed)

    geos = [f"G{i+1}" for i in range(n_geo)]
    brands = [f"B{i+1}" for i in range(n_brand)]

    # Global intercept and true betas
    alpha = 80.0
    b_tv, b_radio, b_social = 2.0, 0.8, 0.4

    # Random intercepts for groups
    a_geo = {g: random.gauss(0, 8.0) for g in geos}
    a_brand = {b: random.gauss(0, 5.0) for b in brands}

    rows = []
    y = []
    for _ in range(n_weeks):
        g = random.choice(geos)
        b = random.choice(brands)
        tv = random.uniform(5.0, 30.0)
        radio = random.uniform(1.0, 10.0)
        social = random.uniform(0.5, 8.0)
        mu = alpha + a_geo[g] + a_brand[b] + b_tv * tv + b_radio * radio + b_social * social
        sales = mu + random.gauss(0, 6.0)
        rows.append({"tv": tv, "radio": radio, "social": social, "geo": g, "brand": b})
        y.append(sales)
    return rows, y, geos, brands, ["tv", "radio", "social"]


def main():
    # 1) Logging and quieting third-party library verbosity
    configure_logging("INFO")
    quiet_third_party_loggers()
    log = get_logger(__name__)
    log.info("Starting hierarchical MMM example")

    # 2) Model Zoo wiring
    model_reg, metric_reg = ModelRegistry(), MetricRegistry()
    model_runner = LibraryModelRunner(max_workers=4, max_active_runs=2, max_retries=1, backoff_initial_s=0.2)
    model_runner.set_run_store(InMemoryRunStore())
    metric_runner = LibraryMetricRunner()
    bind_reg = InMemoryBindingRegistry()
    bus = InMemoryEventBus()
    bus.subscribe("run.status_changed", lambda ev: print("[EVENT] status:", ev.get("status"), ev.get("run_id")))
    bus.subscribe("run.evaluated", lambda ev: print("[EVENT] evaluated:", ev.get("run_id")))
    zoo = ModelZoo(model_reg, metric_reg, model_runner, metric_runner, DefaultParamValidator(), binding_registry=bind_reg, event_bus=bus)

    # 3) Register a hierarchical model spec (hyperparams carry priors, features, etc.)
    hier_hp = HyperParamSpace(fields={
        "draws": HyperParamField("draws", DType.INT, default=500),
        "tune": HyperParamField("tune", DType.INT, default=500),
        "target_accept": HyperParamField("target_accept", DType.FLOAT, default=0.9),
        "feature_names": HyperParamField("feature_names", DType.LIST),
        "geo_key": HyperParamField("geo_key", DType.STR, default="geo"),
        "brand_key": HyperParamField("brand_key", DType.STR, default="brand"),
        "priors": HyperParamField("priors", DType.DICT, default=None),
    })
    model_spec = ModelSpec(
        model_id="pymc_hier_mmm_v1",
        name="BayesianHierarchicalMMM_PyMC",
        version="1.0.0",
        kind=ModelKind.LIBRARY,
        hyperparams=hier_hp,
        tags=["bayesian", "pymc", "hierarchical", "marketing"],
        metadata={"example": True},
    )
    zoo.register_model(model_spec, actor="example")

    # Metrics (R2, MSE)
    r2_spec = EvalMetricSpec("r2_v1", "R2_Score", "1.0.0", MetricDirection.HIGHER_IS_BETTER, MetricKind.LIBRARY)
    mse_spec = EvalMetricSpec("mse_v1", "MeanSquaredError", "1.0.0", MetricDirection.LOWER_IS_BETTER, MetricKind.LIBRARY)
    zoo.register_metric(r2_spec, actor="example")
    zoo.register_metric(mse_spec, actor="example")

    # 4) Register bindings so requests don't need to carry them
    model_binding = ExecutionBinding(
        bound_id="pymc_hier_mmm_v1",
        kind=BindingKind.LIBRARY,
        module="model_zoo.examples.pymc_models",
        callable="pymc_hierarchical_mmm",
    )
    r2_binding = ExecutionBinding("r2_v1", BindingKind.LIBRARY, module="model_zoo.examples.metrics", callable="r2_score")
    mse_binding = ExecutionBinding("mse_v1", BindingKind.LIBRARY, module="model_zoo.examples.metrics", callable="mse")
    bind_reg.register_model_binding("pymc_hier_mmm_v1", model_binding)
    bind_reg.register_metric_binding("r2_v1", r2_binding)
    bind_reg.register_metric_binding("mse_v1", mse_binding)

    # 5) Generate synthetic hierarchical data and wrap in DataRefs
    X_rows, y, geos, brands, feat_names = make_synthetic_marketing_hier()
    X_ref = DataRef(DataRefType.INLINE, inline=X_rows)
    y_ref = DataRef(DataRefType.INLINE, inline=y)

    # Split train/test (75/25)
    n = len(y)
    split = int(0.75 * n)
    train_idx = list(range(split))
    test_idx = list(range(split, n))

    # 6) Custom priors per feature (mix of distributions)
    priors = {
        "beta": {
            # tv: Normal(2,1), radio: HalfNormal(1), social: Gamma(2,1)
            "distribution": ["normal", "half_normal", "gamma"],
            "mu": [2.0, None, None],
            "tau": [1.0, None, None],
            "scale": [None, 1.0, None],
            "shape": [None, None, 2.0],
            "rate": [None, None, 1.0],
        },
        # Observation sigma prior (can switch to gamma/erlang)
        "sigma": {"distribution": "half_normal", "scale": 5.0},
    }

    # 7) Build request with EvaluationSpec
    req = ModelRunRequest(
        model_id="pymc_hier_mmm_v1",
        inputs={"X": X_ref, "y": y_ref},
        params={
            "draws": 400,
            "tune": 400,
            "target_accept": 0.9,
            "chains": 2, "cores": 1, "compute_convergence_checks": True,
            "feature_names": feat_names,
            "geo_key": "geo",
            "brand_key": "brand",
            "priors": priors,
        },
        context=ExecutionContext(random_seed=2025, timeout_s=180),
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

    # 8) Submit and wait
    def wait_for_result(zoo_obj, run_id: str, actor: str = "example", timeout_s: float = 900.0):
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
        log.warning("Hierarchical MMM example skipped: %s", e)
        return

    # 9) Print a compact R-style summary
    print("Hierarchical MMM: status=", res.status.value)
    print(r_style_summary_string(res.summary))


if __name__ == "__main__":
    main()
