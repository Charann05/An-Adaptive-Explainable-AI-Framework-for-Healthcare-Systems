import unittest

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from framework.explanation_generator.generator import format_explanation


class ExplanationFormatterTests(unittest.TestCase):
    def test_formats_signed_contributions_as_percentages(self):
        formatted = format_explanation(
            {
                "prediction": "positive",
                "technique_used": "SHAP",
                "feature_ranking": [
                    {"feature": "supporting", "importance": 3.0},
                    {"feature": "opposing", "importance": -1.0},
                ],
            },
            "Tabular",
        )

        self.assertIn("supporting: +75.0% (supports the prediction)", formatted)
        self.assertIn("opposing: -25.0% (opposes the prediction)", formatted)

    def test_does_not_assign_direction_to_global_feature_importance(self):
        formatted = format_explanation(
            {
                "prediction": 1,
                "technique_used": "FeatureImportance",
                "feature_ranking": [
                    {"feature": "first", "importance": 2.0},
                    {"feature": "second", "importance": 1.0},
                ],
            },
            "Tabular",
        )

        self.assertIn("direction is not provided by this method", formatted)
        self.assertIn("model-level, not specific to this input", formatted)
        self.assertIn("first: 66.7%", formatted)
        self.assertIn("second: 33.3%", formatted)
        self.assertNotIn("supports the prediction", formatted)

    def test_all_zero_scores_do_not_display_invalid_percentages(self):
        formatted = format_explanation(
            {
                "prediction": 1,
                "technique_used": "SHAP",
                "feature_ranking": [
                    {"feature": "first", "importance": 0.0},
                    {"feature": "second", "importance": 0.0},
                ],
            },
            "Tabular",
        )

        self.assertIn("contribution percentages cannot be computed", formatted)
        self.assertNotIn("0.0%", formatted)

    def test_small_nonzero_scores_are_scaled_to_a_full_percentage(self):
        formatted = format_explanation(
            {
                "prediction": 1,
                "technique_used": "SHAP",
                "feature_ranking": [
                    {"feature": "first", "importance": 1e-200},
                    {"feature": "second", "importance": 2e-200},
                ],
            },
            "Tabular",
        )

        self.assertIn("first: +33.3%", formatted)
        self.assertIn("second: +66.7%", formatted)

    def test_image_output_contains_only_prediction_and_heatmap(self):
        formatted = format_explanation(
            {
                "prediction": "pneumonia",
                "technique_used": "GradCAM",
                "feature_ranking": [],
                "raw_output": [[0.1, 0.9]],
            },
            "Image",
            "/tmp/gradcam.png",
        )

        self.assertEqual(
            formatted,
            "Prediction: pneumonia\nHeatmap: /tmp/gradcam.png",
        )


if __name__ == "__main__":
    unittest.main()
