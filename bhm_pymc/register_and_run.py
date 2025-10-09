from model_zoo import (
    ModelZoo, ModelRegistry, MetricRegistry,
    ModelSpec, ModelKind,
    HyperParamSpace, HyperParamField, DType,
    ExecutionBinding, BindingKind,
    DataRef, DataRefType, ExecutionContext,
    ModelRunRequest,
)
from model_zoo.runners.library import LibraryModelRunner, LibraryMetricRunner
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
    
    print(zoo)
    print(dir(zoo))

    hp = HyperParamSpace(fields={
        "draws": HyperParamField("draws", DType.INT, default=1500, required=True),
        "tune": HyperParamField("tune", DType.INT, default=1000, required=True),
    })

    spec = ModelSpec(
        model_id="bhm_pymc_v1",
        name="BayesianHierarchicalSales_PyMC",
        version="1.0.0",
        kind=ModelKind.LIBRARY,
        hyperparams=hp,
        tags=["bayesian", "hierarchical"]
    )
    zoo.register_model(spec, actor="you")

    binding = ExecutionBinding(
        bound_id="bhm_pymc_v1", kind=BindingKind.LIBRARY,
        module="myorg.models.bayes.hierarchical_regression", callable="train_and_infer"
    )

    req = ModelRunRequest(
        model_id="bhm_pymc_v1",
        inputs={
            "X": DataRef(ref_type=DataRefType.URI, uri="s3://bucket/X.parquet", format="parquet"),
            "y": DataRef(ref_type=DataRefType.URI, uri="s3://bucket/y.parquet", format="parquet"),
        },
        params={"draws": 2000, "tune": 1500},
        context=ExecutionContext(random_seed=2025),
        dataset_version="sha256:deadbeef",
        # In a proper app, ModelZoo would pull binding from a BindingRegistry; for demo we attach it directly:
        binding=binding,
    )

    handle = zoo.submit_run(req, actor="you")
    print("RUN:", handle.run_id, handle.status.value)
    res = zoo.get_run_result(handle.run_id, actor="you")
    print("RESULT:", res.status.value, res.predictions.uri if res.predictions else None)

if __name__ == "__main__":
    main()
