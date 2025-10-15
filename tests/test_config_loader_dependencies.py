import copy

from model_zoo.utils.config_loader import load_model_configuration_from_dict


_BASE_CONFIG = {
    "model": {
        "id": "custom_model",
        "binding": {"module": "dummy.module", "callable": "run"},
    },
    "hyperparameters": {
        "schema": [
            {"name": "alpha", "dtype": "float", "default": 0.1},
        ],
        "values": [
            {"model_id": "custom_model", "param": "alpha", "value": "0.1"},
        ],
    },
    "variables": {
        "definitions": [
            {"name": "X", "source": "X"},
        ]
    },
    "context": {}
}


def test_load_configuration_without_dependencies():
    cfg = load_model_configuration_from_dict(copy.deepcopy(_BASE_CONFIG))
    assert cfg.dependencies == []


def test_load_configuration_with_env_dependencies():
    payload = copy.deepcopy(_BASE_CONFIG)
    payload["dependencies"] = [
        "myteam_marketing_mix==1.2.0",
        {"env": "prod", "packages": ["statsmodels>=0.14", "prophet>=1.0"]},
    ]
    cfg_prod = load_model_configuration_from_dict(payload, env="prod")
    assert cfg_prod.dependencies == ["myteam_marketing_mix==1.2.0", "statsmodels>=0.14", "prophet>=1.0"]

    cfg_dev = load_model_configuration_from_dict(payload, env="dev")
    assert cfg_dev.dependencies == ["myteam_marketing_mix==1.2.0"]


def test_load_configuration_with_mapping_dependencies():
    payload = copy.deepcopy(_BASE_CONFIG)
    payload["dependencies"] = {
        "all": ["numpy"],
        "dev": ["pytest"],
        "envs": [
            {"env": "staging", "packages": ["staging-lib"]},
        ],
    }
    cfg_staging = load_model_configuration_from_dict(payload, env="staging")
    assert cfg_staging.dependencies == ["numpy", "staging-lib"]

    cfg_dev = load_model_configuration_from_dict(payload, env="dev")
    assert cfg_dev.dependencies == ["numpy", "pytest"]
