import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from framework.xai_integration.plugins.shap_plugin import SHAPPlugin


class _Adapter:
    framework = "sklearn"

    def __init__(self, framework="sklearn", prediction=None):
        self.framework = framework
        self.prediction = prediction

    def predict(self, model, input_data):
        return self.prediction


class SHAPPluginBehaviorTests(unittest.TestCase):
    def test_selects_output_using_predicted_class_label(self):
        class FakeTreeExplainer:
            def __init__(self, model):
                pass

            def shap_values(self, input_array):
                return np.array([[
                    [0.1, 0.3],
                    [0.2, -0.4],
                ]])

        fake_shap = types.ModuleType("shap")
        fake_shap.TreeExplainer = FakeTreeExplainer
        model = types.SimpleNamespace(
            tree_=object(),
            classes_=np.array([1, 2]),
        )

        with patch.dict(sys.modules, {"shap": fake_shap}):
            result = SHAPPlugin().explain(
                model,
                [[10, 20]],
                {
                    "category": "DecisionTree",
                    "input_type": "Tabular",
                    "feature_names": ["a", "b"],
                },
                adapter=_Adapter(prediction=np.array([2])),
            )

        self.assertEqual(result.prediction, 2)
        self.assertEqual(
            {
                item["feature"]: item["importance"]
                for item in result.feature_ranking
            },
            {"a": 0.3, "b": -0.4},
        )

    def test_gradient_explainer_uses_background_not_input_as_reference(self):
        captured = {}

        class FakeGradientExplainer:
            def __init__(self, model, background):
                captured["background"] = background.copy()

            def shap_values(self, values):
                captured["input"] = values.copy()
                return np.array([[
                    [0.1, 0.2],
                    [0.3, -0.4],
                ]])

        fake_shap = types.ModuleType("shap")
        fake_shap.GradientExplainer = FakeGradientExplainer

        with patch.dict(sys.modules, {"shap": fake_shap}):
            result = SHAPPlugin().explain(
                object(),
                [[1, 1]],
                {
                    "category": "Unknown",
                    "input_type": "Tabular",
                    "background_data": [[0, 0], [2, 2]],
                },
                adapter=_Adapter(
                    framework="tensorflow",
                    prediction=np.array([[0.2, 0.8]]),
                ),
            )

        np.testing.assert_array_equal(captured["background"], [[0, 0], [2, 2]])
        np.testing.assert_array_equal(captured["input"], [[1, 1]])
        self.assertNotEqual(captured["background"].tolist(), [[1, 1]])
        self.assertEqual(result.prediction, 1)

    def test_gradient_explainer_requires_background_data(self):
        fake_shap = types.ModuleType("shap")

        with patch.dict(sys.modules, {"shap": fake_shap}):
            with self.assertRaisesRegex(ValueError, "requires representative"):
                SHAPPlugin().explain(
                    object(),
                    [[1, 1]],
                    {"category": "Unknown", "input_type": "Tabular"},
                    adapter=_Adapter(
                        framework="tensorflow",
                        prediction=np.array([[0.2, 0.8]]),
                    ),
                )


if __name__ == "__main__":
    unittest.main()
