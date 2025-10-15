"""Hierarchical Bayesian marketing mix example using JSON configuration."""
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
from model_zoo.runners.library import LibraryModelRunner, LibraryMetricRunner
from model_zoo.stores.run_store import PostgresRunStore
from model_zoo.utils.config_loader import load_model_configuration, ModelConfiguration
from model_zoo.utils.variable_processing import build_variables
from model_zoo.utils.env import read_env_file, build_postgres_dsn
from model_zoo.utils.logging import configure_logging, quiet_third_party_loggers, get_logger

from model_zoo.examples.pymc_models import pymc_hierarchical_mmm  # noqa: F401
from model_zoo.examples import metrics as example_metrics  # noqa: F401


def ensure_pymc_available() -> bool:
    try:
        import pymc  # noqa: F401
    except Exception as exc:  # pragma: no cover
        print("[WARN] PyMC is required for this example:", exc)
        return False
    return True


def make_synthetic_marketing_hier(n_weeks: int = 52, n_geo: int = 5, n_brand: int = 3, seed: int = 7) -> Tuple[List[Dict[str, Any]], List[float], List[str], List[str], List[str]]:
    import random
    random.seed(seed)
    geos = [f"G{i+1}" for i in range(n_geo)]
    brands = [f"B{i+1}" for i in range(n_brand)]
    alpha = 80.0
    b_tv, b_radio, b_social = 2.0, 0.8, 0.4
    a_geo = {g: random.gauss(0, 8.0) for g in geos}
    a_brand = {b: random.gauss(0, 5.0) for b in brands}
    rows: List[Dict[str, Any]] = []
    y: List[float] = []
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


def configure_zoo(dsn: str, table: str) -> Tuple[ModelZoo, InMemoryBindingRegistry]:
    model_reg, metric_reg = ModelRegistry(), MetricRegistry()
    store = PostgresRunStore(dsn=dsn, table_name=table)
    model_runner = LibraryModelRunner(max_workers=4, max_active_runs=2, max_retries=1, backoff_initial_s=0.2)
    model_runner.set_run_store(store)
    metric_runner = LibraryMetricRunner()
    bind_reg = InMemoryBindingRegistry()
    bus = InMemoryEventBus()
    bus.subscribe("run.status_changed", lambda ev: print("[EVENT] status:", ev.get("status"), ev.get("run_id")))
    bus.subscribe("run.evaluated", lambda ev: print("[EVENT] evaluated:", ev.get("run_id")))
    zoo = ModelZoo(model_reg, metric_reg, model_runner, metric_runner, DefaultParamValidator(), binding_registry=bind_reg, event_bus=bus)
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


def build_data_sources(rows: List[Dict[str, Any]], sales: List[float], geos: List[str], brands: List[str]) -> Dict[str, Any]:
    tv = [row["tv"] for row in rows]
    radio = [row["radio"] for row in rows]
    social = [row["social"] for row in rows]
    return {
        "marketing_rows": rows,
        "sales": sales,
        "tv_spend": tv,
        "radio_spend": radio,
        "social_spend": social,
        "geo_labels": geos,
        "brand_labels": brands,
    }


def build_inputs(cfg: ModelConfiguration, data_sources: Dict[str, Any]) -> Tuple[Dict[str, DataRef], Dict[str, Any]]:
    variables, groups = build_variables(cfg.variables, data_sources)
    input_keys = cfg.model.get("inputs", ["X", "y"])
    inputs: Dict[str, DataRef] = {}
    for key in input_keys:
        value = variables.get(key)
        if value is None:
            raise KeyError(f"Configuration inputs missing key '{key}'")
        inputs[key] = DataRef(DataRefType.INLINE, inline=value)
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


def submit_run(zoo: ModelZoo, cfg: ModelConfiguration, inputs: Dict[str, DataRef], train_idx: List[int], test_idx: List[int], feature_names: List[str]) -> str:
    params = dict(cfg.params)
    params.setdefault("train_indices", train_idx)
    params.setdefault("feature_names", feature_names)
    params.setdefault("geo_key", "geo")
    params.setdefault("brand_key", "brand")
    coeff_payload = cfg.coefficients.to_runner_params()
    for key, value in coeff_payload.items():
        params.setdefault(key, value)
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


def wait_for_result(zoo: ModelZoo, run_id: str, actor: str = "example", timeout_s: float = 1200.0):
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
    parser = argparse.ArgumentParser(description="Config-driven hierarchical PyMC marketing mix example")
    parser.add_argument("--env", default="dev", help="Environment key for configuration entries")
    parser.add_argument("--env-file", default=".env", help="Path to env file with PostgreSQL credentials")
    parser.add_argument("--config", default="configs/marketing_mix_hier.json", help="Path to the hierarchical config")
    parser.add_argument("--weeks", type=int, default=52, help="Number of synthetic weeks")
    parser.add_argument("--geo", type=int, default=5, help="Number of geographies")
    parser.add_argument("--brands", type=int, default=3, help="Number of brands")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--train-fraction", type=float, default=0.75, help="Training split fraction")
    args = parser.parse_args()

    configure_logging("INFO")
    quiet_third_party_loggers()
    log = get_logger(__name__)
    if not ensure_pymc_available():
        log.warning("Skipping example because PyMC is not available")
        return

    env_settings = read_env_file(args.env_file)
    dsn, table_name = build_postgres_dsn(env_settings)

    zoo, bind_reg = configure_zoo(dsn, table_name)
    register_metrics(zoo)
    register_metric_bindings(bind_reg)

    cfg = load_model_configuration(args.config, env=args.env)
    register_model_from_config(zoo, bind_reg, cfg)

    rows, sales, geos, brands, feature_names = make_synthetic_marketing_hier(
        n_weeks=args.weeks,
        n_geo=args.geo,
        n_brand=args.brands,
        seed=args.seed,
    )
    train_split = max(1, min(len(rows), int(len(rows) * args.train_fraction)))
    train_idx = list(range(train_split))
    test_idx = list(range(train_split, len(rows)))

    data_sources = build_data_sources(rows, sales, geos, brands)
    inputs, groups = build_inputs(cfg, data_sources)

    run_id = submit_run(zoo, cfg, inputs, train_idx, test_idx, feature_names)
    result = wait_for_result(zoo, run_id)

    print("Hierarchical run status:", result.status.value)
    print(r_style_summary_string(result.summary))
    print("Variable groups:", list(groups.keys()))


if __name__ == "__main__":
    main()
