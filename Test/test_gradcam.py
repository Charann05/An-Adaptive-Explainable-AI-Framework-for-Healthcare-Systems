import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from adaptive_xai.xai_integration.plugins.gradcam_plugin import GradCAMPlugin
from adaptive_xai.xai_integration.adapters.pytorch_adapter import PyTorchAdapter
from adaptive_xai.xai_integration.adapters.tensorflow_adapter import TensorFlowAdapter

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
                return np.array([[0.1, 0.9]])

            def predict_proba(self, model, input_data):
                raise AssertionError("Grad-CAM should normalize raw model output.")

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

    def test_single_logit_binary_prediction_selects_correct_cam_class(self):
        class FakeAdapter:
            framework = "test"

            def supports_gradients(self):
                return True

            def get_last_conv_layer(self, model):
                return "target-layer", "conv"

            def predict(self, model, input_data):
                return np.array([[-0.7]])

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

        self.assertEqual(result.prediction, 0)
        self.assertAlmostEqual(result.confidence_score, 66.8, places=1)
        self.assertEqual(adapter.target_prediction, 0)

    def test_pytorch_gradcam_target_handles_single_logit_binary_models(self):
        class FakeGradCAM:
            def __init__(self, *, model, target_layers):
                self.model = model
                self.target_layers = target_layers

            def __call__(self, *, input_tensor, targets):
                return targets[0](np.array([[-0.7]]))

        fake_cam_module = SimpleNamespace(GradCAM=FakeGradCAM)
        adapter = PyTorchAdapter()
        with patch.dict("sys.modules", {"pytorch_grad_cam": fake_cam_module}):
            positive_class_score = adapter.run_gradcam(
                object(),
                np.zeros((1, 3, 8, 8)),
                object(),
                1,
            )
            negative_class_score = adapter.run_gradcam(
                object(),
                np.zeros((1, 3, 8, 8)),
                object(),
                0,
            )

        self.assertEqual(positive_class_score.tolist(), [-0.7])
        self.assertEqual(negative_class_score.tolist(), [0.7])

    def test_tensorflow_gradcam_targets_both_sides_of_single_logit_output(self):
        class FakeTensor(np.ndarray):
            def numpy(self):
                return np.asarray(self)

        def fake_maximum(left, right):
            return np.maximum(left, right).view(FakeTensor)

        class FakeGradientTape:
            observed_loss = None

            def __enter__(self):
                return self

            def __exit__(self, exception_type, exception, traceback):
                return False

            def gradient(self, loss, conv_output):
                self.observed_loss = loss.copy()
                return np.ones_like(conv_output)

        class FakeGradModel:
            def __call__(self, input_data, *, training):
                conv_output = np.ones((1, 2, 2, 1))
                predictions = np.array([[-0.7]])
                return conv_output, predictions

        tape = FakeGradientTape()
        fake_tensorflow = SimpleNamespace(
            keras=SimpleNamespace(
                models=SimpleNamespace(
                    Model=lambda **kwargs: FakeGradModel(),
                ),
            ),
            GradientTape=lambda: tape,
            reduce_mean=np.mean,
            newaxis=np.newaxis,
            maximum=fake_maximum,
            squeeze=np.squeeze,
            math=SimpleNamespace(reduce_max=np.max),
        )
        adapter = TensorFlowAdapter()
        model = SimpleNamespace(inputs=object(), output=object())
        with patch.dict("sys.modules", {"tensorflow": fake_tensorflow}):
            adapter.run_gradcam(
                model,
                np.zeros((1, 2, 2, 3)),
                SimpleNamespace(output=SimpleNamespace()),
                0,
            )
            negative_class_loss = tape.observed_loss.copy()
            adapter.run_gradcam(
                model,
                np.zeros((1, 2, 2, 3)),
                SimpleNamespace(output=SimpleNamespace()),
                1,
            )
            positive_class_loss = tape.observed_loss.copy()

        self.assertEqual(negative_class_loss.tolist(), [0.7])
        self.assertEqual(positive_class_loss.tolist(), [-0.7])

    def test_explains_convolutional_model(self):
        if not MODEL_PATH.is_file():
            self.skipTest(f"Model fixture not found: {MODEL_PATH}")

        try:
            import torch
            import pytorch_grad_cam  # noqa: F401
        except ImportError as error:
            self.skipTest(f"Grad-CAM dependencies are unavailable: {error}")

        from adaptive_xai.xai_integration.engine import run_explanation

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