from model_zoo.runners.base import ModelRunner, MetricRunner
import inspect

def test_runner_is_abstract():
    assert inspect.isabstract(ModelRunner)
    assert inspect.isabstract(MetricRunner)
