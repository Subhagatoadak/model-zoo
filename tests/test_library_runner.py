from model_zoo.contracts.run_requests import ModelRunRequest
from model_zoo.contracts.io import DataRef, DataRefType
from model_zoo.bindings.base import ExecutionBinding, BindingKind
from model_zoo.runners.library import LibraryModelRunner


def test_library_runner_simple_linreg():
    X = [i / 10 for i in range(1, 51)]
    y = [1.0 + 2.0 * x for x in X]

    binding = ExecutionBinding(
        bound_id="linear_regression_v1",
        kind=BindingKind.LIBRARY,
        module="model_zoo.examples.models",
        callable="linreg_train_and_infer",
    )

    req = ModelRunRequest(
        model_id="linear_regression_v1",
        inputs={
            "X": DataRef(ref_type=DataRefType.INLINE, inline=X),
            "y": DataRef(ref_type=DataRefType.INLINE, inline=y),
        },
        params={"fit_intercept": True, "feature_name": "X"},
        binding=binding,
    )

    runner = LibraryModelRunner()
    handle = runner.submit(req)

    # Synchronous execution in our runner; result should be ready
    # Poll until finished (fast path)
    import time
    for _ in range(100):
        res = runner.result(handle.run_id)
        if res.status.name != "STARTED" and not res.metadata.get("in_progress"):
            break
        time.sleep(0.01)
    assert res.status.name == "SUCCEEDED"
    assert res.predictions is not None
    assert len(res.predictions.inline) == len(X)
    assert isinstance(res.diagnostics, dict)
    assert any(k.startswith("coef[") for k in res.diagnostics.keys())
