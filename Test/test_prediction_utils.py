import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

if __package__:
    from . import _bootstrap
else:
    import _bootstrap


import numpy as np

from adaptive_xai.xai_integration.adapters.tensorflow_adapter import (
    TensorFlowAdapter,
)
from adaptive_xai.xai_integration.adapters.pytorch_adapter import PyTorchAdapter
from adaptive_xai.xai_integration.prediction_utils import (
    predicted_class_and_confidence,
    predicted_value,
)


class PredictedValueTests(unittest.TestCase):
    def test_tensorflow_adapter_unwraps_single_tensor_for_single_input(self):
        tensor = np.zeros((1, 224, 224, 3))

        class FakeModel:
            input = type("SymbolicInput", (), {
                "shape": (None, 224, 224, 3),
            })()

            def __call__(self, value, *, training):
                self.received = value
                self.training = training
                return value

        model = FakeModel()
        TensorFlowAdapter().predict(model, [tensor])

        self.assertIs(model.received, tensor)
        self.assertFalse(model.training)

    def test_tensorflow_adapter_wraps_input_for_singleton_input_list(self):
        symbolic_input = object()
        tensor = np.zeros((50, 224, 224, 3))

        class FakeModel:
            input = [symbolic_input]

            def __call__(self, value, *, training):
                self.received = value
                return value

        model = FakeModel()
        TensorFlowAdapter().predict(model, tensor)

        self.assertEqual(len(model.received), 1)
        self.assertIs(model.received[0], tensor)

    def test_tensorflow_adapter_preserves_tabular_batch_shape(self):
        input_data = [[1.0, 2.0]]

        class FakeModel:
            input = object()

            def __call__(self, value, *, training):
                self.received = value
                return value

        model = FakeModel()
        TensorFlowAdapter().predict(model, input_data)

        self.assertIs(model.received, input_data)

    def test_tensorflow_adapter_preserves_single_sample_batch(self):
        sample = np.array([1.0, 2.0])
        input_data = [sample]

        class FakeModel:
            input = type("SymbolicInput", (), {"shape": (None, 2)})()

            def __call__(self, value, *, training):
                self.received = value
                return value

        model = FakeModel()
        TensorFlowAdapter().predict(model, input_data)

        self.assertIs(model.received, input_data)

    def test_tensorflow_predict_proba_normalizes_binary_logits(self):
        class FakeModel:
            input = None

            def __call__(self, value, *, training):
                return np.array([[-0.7]])

        probabilities = TensorFlowAdapter().predict_proba(
            FakeModel(),
            np.zeros((1, 224, 224, 3)),
        )

        np.testing.assert_allclose(
            probabilities,
            [[0.66818777, 0.33181223]],
            rtol=1e-6,
        )

    def test_uses_argmax_for_one_dimensional_class_scores(self):
        self.assertEqual(
            predicted_value(
                np.array([0.1, 0.8, 0.1]),
                "Classification",
                class_scores=True,
            ),
            1,
        )

    def test_uses_argmax_for_batched_class_scores(self):
        self.assertEqual(
            predicted_value(
                np.array([[0.1, 0.2, 0.7]]),
                "Classification",
                class_scores=True,
            ),
            2,
        )

    def test_preserves_one_dimensional_class_labels(self):
        self.assertEqual(
            predicted_value(
                np.array([2]),
                "Classification",
            ),
            2,
        )

    def test_preserves_regression_output(self):
        self.assertEqual(
            predicted_value(
                np.array([0.7, 0.2]),
                "Regression",
                class_scores=True,
            ),
            0.7,
        )

    def test_interprets_single_logit_as_binary_class_score(self):
        prediction, confidence = predicted_class_and_confidence(
            np.array([[-0.7]])
        )

        self.assertEqual(prediction, 0)
        self.assertAlmostEqual(confidence, 0.6682, places=4)

    def test_interprets_single_probability_as_binary_class_score(self):
        prediction, confidence = predicted_class_and_confidence(
            np.array([[0.8]])
        )

        self.assertEqual(prediction, 1)
        self.assertAlmostEqual(confidence, 0.8)

    def test_pytorch_adapter_converts_lime_numpy_batch_and_normalizes_output(self):
        class FakeTensor:
            def __init__(self, value, dtype="float32", device="cpu"):
                self.value = np.asarray(value)
                self.dtype = dtype
                self.device = device

            def to(self, *, device=None, dtype=None):
                return FakeTensor(
                    self.value,
                    dtype=dtype or self.dtype,
                    device=device or self.device,
                )

            def is_floating_point(self):
                return True

        class FakeParameter:
            dtype = "model-float"
            device = "accelerator"

            def is_floating_point(self):
                return True

        class FakeModel:
            def __init__(self):
                self.received = None

            def parameters(self):
                return iter([FakeParameter()])

            def eval(self):
                pass

            def __call__(self, value):
                self.received = value
                return np.array([[-0.7]])

        fake_torch = SimpleNamespace(
            Tensor=FakeTensor,
            float32="float32",
            as_tensor=lambda value, dtype: FakeTensor(value, dtype=dtype),
            no_grad=nullcontext,
        )
        model = FakeModel()
        with patch.dict("sys.modules", {"torch": fake_torch}):
            probabilities = PyTorchAdapter().predict_proba(
                model,
                np.zeros((1, 3, 8, 8)),
            )

        self.assertIsInstance(model.received, FakeTensor)
        self.assertEqual(model.received.dtype, "model-float")
        self.assertEqual(model.received.device, "accelerator")
        np.testing.assert_allclose(
            probabilities,
            [[0.66818777, 0.33181223]],
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
