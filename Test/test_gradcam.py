import unittest
from pathlib import Path

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

MODEL_PATH = Path(__file__).with_name("test_cnn_model.pt")


class GradCAMIntegrationTests(unittest.TestCase):
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