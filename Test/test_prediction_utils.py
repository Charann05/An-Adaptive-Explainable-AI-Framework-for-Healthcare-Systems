import unittest

if __package__:
    from . import _bootstrap
else:
    import _bootstrap


import numpy as np

from framework.xai_integration.prediction_utils import predicted_value


class PredictedValueTests(unittest.TestCase):
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
