from __future__ import annotations

from model_zoo.utils.config_loader import load_model_configuration_from_dict


def test_legacy_payload_is_upgraded_to_canonical_schema() -> None:
    variable_csv = """COLUMN,CATEGORIZATION,VARIABLE LABEL,INCLUDE,DECOMP TYPE,SIGN EXPECTATION,LOWER BOUND,UPPER BOUND,PRIOR TYPE,PRIOR WEIGHT,PRIOR
ENDPERIOD,DATE,ENDPERIOD,True,,,,,,,
DEPENDENT_VARIABLE,RESPONSE,DEPENDENT VARIABLE,False,,,,,,,
OLV_YT23_MEDIA,MEDIA,OLV_YT23_MEDIA,True,,POSITIVE,0,inf,normal,1.5,0.25
"""
    legacy_payload = {
        "algorithm": {"type": "least_squares"},
        "errorallocation": {"activate": True},
        "spec": {
            "time_dim": "ENDPERIOD",
            "dep_var": "DEPENDENT_VARIABLE",
            "prior": True,
        },
        "params": {
            "LeastSquaresAlgorithm": {
                "CoefficientStatistics": True,
            },
            "GLMVSgibbsAlgorithm": {
                "MinIterations": 2000,
            },
        },
        "variable_config": variable_csv,
    }

    cfg = load_model_configuration_from_dict(legacy_payload, env="dev", model_id="legacy_test_model")

    assert cfg.model["id"] == "legacy_test_model"
    assert cfg.model["kind"] == "pipeline"
    assert cfg.model["name"] == "Least Squares Model"
    hyperparams = cfg.hyperparam_space.fields
    assert "LeastSquaresAlgorithm.CoefficientStatistics" in hyperparams
    assert "GLMVSgibbsAlgorithm.MinIterations" in hyperparams
    assert cfg.params["LeastSquaresAlgorithm.CoefficientStatistics"] is True
    assert cfg.params["GLMVSgibbsAlgorithm.MinIterations"] == 2000

    definitions = cfg.variables.definitions
    assert "OLV_YT23_MEDIA" in definitions
    olv_meta = definitions["OLV_YT23_MEDIA"].metadata
    assert olv_meta.get("sign_expectation") == "POSITIVE"

    assert cfg.coefficients.intercept is None
    olv_coeff = cfg.coefficients.variables["OLV_YT23_MEDIA"]
    assert olv_coeff.distribution == "normal"
    assert olv_coeff.params.get("mu") == 0.25
    assert olv_coeff.params.get("sigma") == 1.5
    assert olv_coeff.constraints.get("sign") == "positive"

    legacy_context = cfg.context.get("legacy", {})
    assert legacy_context["algorithm"]["type"] == "least_squares"
    assert "original_payload" in legacy_context
