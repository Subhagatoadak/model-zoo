from model_zoo.bindings.base import ExecutionBinding, BindingKind

def test_binding_creation():
    b = ExecutionBinding(bound_id="m1", kind=BindingKind.API, endpoint="https://example.com", method="POST")
    assert b.kind is BindingKind.API
