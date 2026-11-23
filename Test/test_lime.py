import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from adaptive_xai.xai_integration.plugins.lime_plugin import LIMEPlugin

MODEL_PATH = Path(__file__).with_name("test_rf_model.pkl")


class LIMEIntegrationTests(unittest.TestCase):
    def test_tabular_predictions_keep_class_columns_stable(self):
        import numpy as np

        observed = {}

        class FakeExplanation:
            def as_list(self):
                return [
                    ("age > 61.00", 0.1),
                    ("203.25 < chol <= 237.50", 0.2),
                    ("ca <= 0.00", 0.3),
                ]

        class FakeLimeTabularExplainer:
            def __init__(self, **kwargs):
                pass

            def explain_instance(self, instance, predict_fn, *, num_features):
                observed["probabilities"] = predict_fn(
                    np.array([[0.2, 0.3], [0.4, 0.5]])
                )
                return FakeExplanation()

        class FakeAdapter:
            framework = "sklearn"

            def predict_proba(self, model, input_data):
                raise NotImplementedError

            def predict(self, model, input_data):
                return np.zeros(len(input_data), dtype=int)

        fake_lime = SimpleNamespace(
            lime_tabular=SimpleNamespace(
                LimeTabularExplainer=FakeLimeTabularExplainer,
            ),
        )
        model = SimpleNamespace(classes_=np.array([0, 1]))

        with patch.dict(
            "sys.modules",
            {
                "lime": fake_lime,
                "lime.lime_tabular": fake_lime.lime_tabular,
            },
        ):
            result = LIMEPlugin()._explain_tabular(
                model,
                [[0.1, 0.2]],
                {
                    "input_type": "Tabular",
                    "task": "Classification",
                    "feature_names": ["a", "b"],
                    "background_data": [[0.0, 0.0], [1.0, 1.0]],
                },
                FakeAdapter(),
            )

        np.testing.assert_array_equal(
            observed["probabilities"],
            [[1.0, 0.0], [1.0, 0.0]],
        )
        self.assertEqual(result.prediction, 0)
        self.assertEqual(
            [item["feature"] for item in result.feature_ranking],
            [
                "age",
                "chol",
                "ca",
            ],
        )

        from adaptive_xai.explanation_generator.generator import (
            format_xai_output,
        )

        output = format_xai_output(
            {
                "technique_used": result.technique_used,
                "prediction": result.prediction,
                "feature_ranking": result.feature_ranking,
            },
            "Tabular",
        )
        self.assertNotRegex(output, r"[<>]=?")

        from adaptive_xai.xai_integration.visualization import (
            display_tabular_attributions,
        )

        observed = {}

        class FakeAxis:
            def barh(self, *args, **kwargs):
                pass

            def set_yticks(self, positions, *, labels):
                observed["labels"] = labels

            def invert_yaxis(self):
                pass

            def axvline(self, *args, **kwargs):
                pass

            def set_xlabel(self, *args, **kwargs):
                pass

            def set_title(self, *args, **kwargs):
                pass

            def grid(self, *args, **kwargs):
                pass

            def set_axisbelow(self, *args, **kwargs):
                pass

        class FakeFigure:
            def tight_layout(self):
                pass

            def savefig(self, *args, **kwargs):
                pass

        fake_matplotlib = ModuleType("matplotlib")
        fake_matplotlib.__path__ = []
        fake_matplotlib.get_backend = lambda: "agg"
        fake_pyplot = ModuleType("matplotlib.pyplot")
        fake_pyplot.subplots = lambda **kwargs: (FakeFigure(), FakeAxis())
        fake_pyplot.close = lambda figure: None
        fake_matplotlib.pyplot = fake_pyplot
        with patch.dict(
            "sys.modules",
            {
                "matplotlib": fake_matplotlib,
                "matplotlib.pyplot": fake_pyplot,
            },
        ):
            display_tabular_attributions(
                result.feature_ranking,
                "unused.png",
                "Feature contributions",
            )
        self.assertEqual(observed["labels"], ["age", "chol", "ca"])

    def test_tabular_lime_wraps_samples_as_dataframe_for_named_models(self):
        import numpy as np
        import pandas as pd

        observed = []

        class FakeExplanation:
            def as_list(self):
                return [("age > 30", 0.1)]

        class FakeLimeTabularExplainer:
            def __init__(self, *, training_data, feature_names, **kwargs):
                self.feature_names = feature_names
                self.training_data = training_data

            def explain_instance(self, instance, predict_fn, *, num_features):
                predict_fn(np.array([[20, 120], [30, 130]]))
                return FakeExplanation()

        class FakeAdapter:
            framework = "sklearn"

            def predict_proba(self, model, input_data):
                observed.append(input_data)
                return np.tile([[0.8, 0.2]], (len(input_data), 1))

            def predict(self, model, input_data):
                observed.append(input_data)
                return np.zeros(len(input_data), dtype=int)

        fake_lime = SimpleNamespace(
            lime_tabular=SimpleNamespace(
                LimeTabularExplainer=FakeLimeTabularExplainer,
            ),
        )
        model = SimpleNamespace(
            feature_names_in_=np.array(["age", "glucose"]),
            classes_=np.array([0, 1]),
        )
        with patch.dict(
            "sys.modules",
            {
                "lime": fake_lime,
                "lime.lime_tabular": fake_lime.lime_tabular,
            },
        ):
            result = LIMEPlugin()._explain_tabular(
                model,
                [[25, 125]],
                {
                    "input_type": "Tabular",
                    "task": "Classification",
                    "background_data": [[20, 120], [30, 130]],
                },
                FakeAdapter(),
            )

        self.assertEqual(result.prediction, 0)
        self.assertEqual(len(observed), 2)
        self.assertTrue(all(isinstance(value, pd.DataFrame) for value in observed))
        self.assertEqual(list(observed[0].columns), ["age", "glucose"])
        self.assertEqual(list(observed[1].columns), ["age", "glucose"])

    def test_image_explainer_uses_channels_first_for_pytorch_batches(self):
        import numpy as np

        observed = {}

        class FakeExplanation:
            top_labels = [1]

            def get_image_and_mask(
                self,
                label,
                *,
                positive_only,
                num_features,
            ):
                return np.zeros((4, 4, 3)), np.ones((4, 4))

        class FakeLimeImageExplainer:
            def explain_instance(
                self,
                image,
                predict_fn,
                *,
                top_labels,
                num_samples,
            ):
                observed["image_shape"] = image.shape
                observed["predictions"] = predict_fn(
                    np.stack([image, image])
                )
                return FakeExplanation()

        class FakeAdapter:
            framework = "pytorch"

            def predict_proba(self, model, input_data):
                observed["model_input_shape"] = input_data.shape
                return np.tile([[0.2, 0.8]], (len(input_data), 1))

        fake_lime = SimpleNamespace(
            lime_image=SimpleNamespace(
                LimeImageExplainer=FakeLimeImageExplainer,
            ),
        )
        input_data = np.zeros((1, 3, 4, 4), dtype=np.float32)
        with patch.dict(
            "sys.modules",
            {
                "lime": fake_lime,
                "lime.lime_image": fake_lime.lime_image,
            },
        ):
            result = LIMEPlugin()._explain_image(
                object(),
                input_data,
                {"input_type": "Image"},
                FakeAdapter(),
            )

        self.assertEqual(observed["image_shape"], (4, 4, 3))
        self.assertEqual(observed["model_input_shape"], (2, 3, 4, 4))
        np.testing.assert_allclose(observed["predictions"], [[0.2, 0.8]] * 2)
        self.assertEqual(result.prediction, 1)

    def test_explains_random_forest_model(self):
        if not MODEL_PATH.is_file():
            self.skipTest(f"Model fixture not found: {MODEL_PATH}")

        try:
            import joblib
            import lime  # noqa: F401
        except ImportError as error:
            self.skipTest(f"LIME dependencies are unavailable: {error}")

        from adaptive_xai.xai_integration.engine import run_explanation
        model = joblib.load(MODEL_PATH)
        result = run_explanation(
            model=model,
            input_data=[[5.1, 3.5, 1.4, 0.2]],
            model_info={
                "category": "RandomForest",
                "input_type": "Tabular",
                "background_data": [
                    [5.1, 3.5, 1.4, 0.2],
                    [6.2, 3.4, 5.4, 2.3],
                    [5.9, 3.0, 5.1, 1.8],
                ],
                "framework": "sklearn",
            },
            selector_result={"selected": "LIME", "alternatives": []},
        )
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()