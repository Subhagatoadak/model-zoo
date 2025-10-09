from model_zoo.registry.models import ModelSpec, ModelRegistry, ModelKind
from model_zoo.contracts.hyperparams import HyperParamSpace, HyperParamField, DType

def test_model_registry_register_get():
    reg = ModelRegistry()
    hp = HyperParamSpace(fields={"alpha": HyperParamField("alpha", DType.FLOAT, default=0.1)})
    spec = ModelSpec("m1", "Model One", "1.0.0", ModelKind.LIBRARY, hp)
    reg.register(spec)
    got = reg.get("m1")
    assert got.name == "Model One"
