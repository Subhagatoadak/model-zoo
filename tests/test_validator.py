import pytest
from model_zoo.contracts.validators import DefaultParamValidator
from model_zoo.registry.models import ModelSpec, ModelKind
from model_zoo.contracts.hyperparams import HyperParamSpace, HyperParamField, DType


def test_default_param_validator_types_and_required():
    v = DefaultParamValidator()
    space = HyperParamSpace(fields={
        "a": HyperParamField("a", DType.INT, required=True),
        "b": HyperParamField("b", DType.FLOAT, min_value=0, max_value=10),
        "c": HyperParamField("c", DType.STR, pattern=r"^x.*"),
    })
    spec = ModelSpec(model_id="m", name="M", version="1", kind=ModelKind.LIBRARY, hyperparams=space)

    # Missing required
    with pytest.raises(Exception):
        v.validate_model_params(spec, {"b": 1.0, "c": "xok"})

    # Wrong type and pattern
    with pytest.raises(Exception):
        v.validate_model_params(spec, {"a": 1, "b": 1.0, "c": "bad"})

    # Valid
    v.validate_model_params(spec, {"a": 2, "b": 5.5, "c": "xgood", "extra": True})

