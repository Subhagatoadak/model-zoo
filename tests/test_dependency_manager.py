import sys

import pytest

from model_zoo.utils.dependency_manager import DependencyManager, DependencyResolutionError


def test_dependency_manager_normalize_and_cache(tmp_path, monkeypatch):
    manager = DependencyManager(base_dir=tmp_path)
    monkeypatch.setenv("MODEL_ZOO_SKIP_DEP_INSTALL", "1")

    env = manager.ensure(["statsmodels>=0.14", "statsmodels>=0.14", "   numpy>=1.26   "])

    assert env is not None
    assert list(env.requirements) == ["statsmodels>=0.14", "numpy>=1.26"]
    marker = env.location / ".installed"
    assert marker.exists()

    # Ensure sys.path is updated idempotently
    try:
        manager.apply(env)
        assert str(env.location) in sys.path

        manager.apply(env)
        assert sys.path.count(str(env.location)) == 1
    finally:
        while str(env.location) in sys.path:
            sys.path.remove(str(env.location))

    monkeypatch.delenv("MODEL_ZOO_SKIP_DEP_INSTALL", raising=False)


def test_dependency_manager_invalid_spec(tmp_path):
    manager = DependencyManager(base_dir=tmp_path)
    with pytest.raises(DependencyResolutionError):
        manager.normalize(["not a valid @@ requirement"])
