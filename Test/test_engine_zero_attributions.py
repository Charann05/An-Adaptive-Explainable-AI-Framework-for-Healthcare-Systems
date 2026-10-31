import unittest
from types import SimpleNamespace
from unittest.mock import patch

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from adaptive_xai.xai_integration.engine import XAIIntegrationEngine
from adaptive_xai.xai_integration.output_schema import ExplanationResult


class _FakePlugin:
    def __init__(self, name, feature_ranking):
        self.name = name
        self.feature_ranking = feature_ranking

    def is_compatible(self, model_info):
        return True

    def explain(self, **kwargs):
        return ExplanationResult(
            technique_used=self.name,
            prediction=1,
            feature_ranking=self.feature_ranking,
        )


class EngineZeroAttributionTests(unittest.TestCase):
    def test_tries_local_explainer_before_global_importance_on_zero_scores(self):
        plugins = {
            "SHAP": _FakePlugin("SHAP", [
                {"feature": "a", "importance": 0.0},
                {"feature": "b", "importance": 0.0},
            ]),
            "LIME": _FakePlugin("LIME", [
                {"feature": "a", "importance": 0.2},
                {"feature": "b", "importance": -0.1},
            ]),
            "FeatureImportance": _FakePlugin("FeatureImportance", [
                {"feature": "a", "importance": 0.8},
                {"feature": "b", "importance": 0.2},
            ]),
        }
        attempted = []

        def get_plugin(name):
            attempted.append(name)
            return plugins[name]

        with (
            patch(
                "adaptive_xai.xai_integration.engine.resolve_adapter",
                return_value=object(),
            ),
            patch(
                "adaptive_xai.xai_integration.engine.prepare_dependencies",
                return_value=SimpleNamespace(available=True),
            ),
            patch("adaptive_xai.xai_integration.engine.get_plugin", side_effect=get_plugin),
        ):
            result = XAIIntegrationEngine().explain(
                model=object(),
                input_data=[[1, 2]],
                model_info={"input_type": "Tabular"},
                selector_result={
                    "selected": "SHAP",
                    "alternatives": ["FeatureImportance", "LIME"],
                },
            )

        self.assertEqual(result.technique_used, "LIME")
        self.assertEqual(attempted, ["SHAP", "LIME"])
        self.assertEqual(result.metadata["fallback_details"][0]["stage"], "attribution")


if __name__ == "__main__":
    unittest.main()
