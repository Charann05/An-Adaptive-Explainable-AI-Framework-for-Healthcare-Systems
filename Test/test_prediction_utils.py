import unittest

if __package__:
    from . import _bootstrap
else:
    import _bootstrap


import numpy as np

from framework.xai_integration.adapters.tensorflow_adapter import (
    TensorFlowAdapter,
)
from framework.xai_integration.prediction_utils import predicted_value


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


if __name__ == "__main__":
    unittest.main()
