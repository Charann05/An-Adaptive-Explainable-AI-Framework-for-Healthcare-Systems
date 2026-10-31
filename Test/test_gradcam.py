import unittest
from pathlib import Path

import numpy as np

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from framework.xai_integration.plugins.gradcam_plugin import GradCAMPlugin

MODEL_PATH = Path(__file__).with_name("test_cnn_model.pt")


class GradCAMIntegrationTests(unittest.TestCase):
    def test_uses_one_prediction_result_to_select_cam_target(self):
        class FakeAdapter:
            framework = "test"

            def supports_gradients(self):
                return True

            def get_last_conv_layer(self, model):
                return "target-layer", "conv"

            def predict(self, model, input_data):
                raise AssertionError("Grad-CAM should not make an unused prediction.")

            def predict_proba(self, model, input_data):
                return np.array([[0.1, 0.9]])

            def run_gradcam(self, model, input_data, target_layer, prediction):
                self.target_prediction = prediction
                return np.ones((8, 8))

        adapter = FakeAdapter()
        result = GradCAMPlugin().explain(
            model=object(),
            input_data=np.zeros((1, 3, 8, 8)),
            model_info={"category": "CNN", "input_type": "Image"},
            adapter=adapter,
        )

        self.assertEqual(adapter.target_prediction, 1)
        self.assertEqual(result.prediction, 1)
        self.assertEqual(result.confidence_score, 90.0)

    def test_explains_convolutional_model(self):
        if not MODEL_PATH.is_file():
            self.skipTest(f"Model fixture not found: {MODEL_PATH}")

        try:
            import torch
            import pytorch_grad_cam  # noqa: F401
        except ImportError as error:
            self.skipTest(f"Grad-CAM dependencies are unavailable: {error}")

        from framework.xai_integration.engine import run_explanation

        model = torch.load(MODEL_PATH, weights_only=False)
        result = run_explanation(
            model=model,
            input_data=torch.rand(1, 3, 32, 32),
            model_info={"category": "CNN", "input_type": "Image"},
            selector_result={
                "selected": "GradCAM",
                "alternatives": ["SHAP", "LIME"],
            },
        )
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()