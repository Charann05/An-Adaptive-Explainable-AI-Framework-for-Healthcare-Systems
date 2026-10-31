import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from framework.xai_integration.plugins.lime_plugin import LIMEPlugin

MODEL_PATH = Path(__file__).with_name("test_rf_model.pkl")


class LIMEIntegrationTests(unittest.TestCase):
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

        from framework.xai_integration.engine import run_explanation
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