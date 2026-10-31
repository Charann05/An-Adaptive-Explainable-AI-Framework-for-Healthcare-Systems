import tempfile
import unittest
from pathlib import Path

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from framework.explanation_generator.generator import (
    format_explanation,
    format_xai_output,
)
from framework.explanation_generator.template_engine import TemplateEngine


class ExplanationFormatterTests(unittest.TestCase):
    def test_xai_values_and_human_explanation_are_separate(self):
        result = {
            "prediction": "positive",
            "technique_used": "SHAP",
            "feature_ranking": [
                {"feature": "supporting", "importance": 3.0},
                {"feature": "opposing", "importance": -1.0},
            ],
        }
        xai_output = format_xai_output(result, "Tabular")
        explanation = format_explanation(result, "Tabular")

        self.assertIn("supporting: +3", xai_output)
        self.assertIn("relative magnitude: 75.0%", xai_output)
        self.assertIn("opposing: -1", xai_output)
        self.assertNotIn("supported the predicted outcome", xai_output)
        self.assertIn("In simple terms", explanation)
        self.assertIn(
            "supporting was the biggest clue, and it pointed the model toward this answer.",
            explanation,
        )
        self.assertIn(
            "opposing was another important clue, and it pointed the model away from this answer.",
            explanation,
        )
        self.assertNotIn("75.0%", explanation)
        self.assertIn(
            "do not prove that a feature caused the result",
            explanation,
        )
        self.assertIn("not a medical diagnosis", explanation)

    def test_global_importance_is_not_described_as_per_input_or_directional(self):
        formatted = format_explanation(
            {
                "prediction": "positive",
                "technique_used": "FeatureImportance",
                "feature_ranking": [
                    {"feature": "glucose", "importance": 2.0},
                    {"feature": "age", "importance": 1.0},
                ],
            },
            "Tabular",
        )

        self.assertIn("glucose, age", formatted)
        self.assertIn("Across many decisions", formatted)
        self.assertIn("not a reason for this particular answer", formatted)
        self.assertNotIn("supported the predicted outcome", formatted)

    def test_zero_scores_do_not_create_a_directional_explanation(self):
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

        self.assertIn("not enough information to explain", formatted)
        self.assertNotIn("supported the predicted outcome", formatted)

    def test_small_scores_are_scaled_only_in_the_xai_output(self):
        result = {
            "prediction": 1,
            "technique_used": "SHAP",
            "feature_ranking": [
                {"feature": "first", "importance": 1e-200},
                {"feature": "second", "importance": 2e-200},
            ],
        }
        xai_output = format_xai_output(result, "Tabular")
        explanation = format_explanation(result, "Tabular")

        self.assertIn("relative magnitude: 33.3%", xai_output)
        self.assertIn("relative magnitude: 66.7%", xai_output)
        self.assertNotIn("%", explanation)

    def test_image_visualization_is_separate_from_plain_language_explanation(self):
        result = {
            "prediction": "pneumonia",
            "technique_used": "GradCAM",
            "confidence_score": 92.4,
            "feature_ranking": [],
            "raw_output": [[0.1, 0.9]],
        }
        xai_output = format_xai_output(
            result,
            "Image",
            "/tmp/gradcam.png",
        )
        explanation = format_explanation(
            result,
            "Image",
            "/tmp/gradcam.png",
        )

        self.assertIn("Grad-CAM visualization: /tmp/gradcam.png", xai_output)
        self.assertNotIn("Grad-CAM visualization", explanation)
        self.assertIn("In simple terms", explanation)
        self.assertIn("A highlight does not mean that area caused", explanation)
        self.assertIn("score of 92.4%", explanation)

    def test_confidence_is_reported_as_model_score_not_clinical_certainty(self):
        formatted = format_explanation(
            {
                "prediction": "pneumonia",
                "technique_used": "GradCAM",
                "confidence_score": 92.4,
                "feature_ranking": [],
                "raw_output": [[0.1, 0.9]],
            },
            "Image",
        )

        self.assertIn(
            "does not tell us how certain the answer is in real life",
            formatted,
        )

    def test_tree_interpreter_has_a_local_attribution_explanation(self):
        result = {
            "prediction": 1,
            "technique_used": "TreeInterpreter",
            "feature_ranking": [
                {"feature": "age", "importance": 0.8},
                {"feature": "glucose", "importance": -0.2},
            ],
        }

        xai_output = format_xai_output(result, "Tabular")
        explanation = format_explanation(result, "Tabular")

        self.assertIn("Tree-interpreter feature contributions", xai_output)
        self.assertIn("features that mattered most", explanation)
        self.assertIn("age was the biggest clue", explanation)
        self.assertIn("do not prove that a feature caused", explanation)

    def test_template_engine_loads_and_caches_json_templates(self):
        with tempfile.TemporaryDirectory() as directory:
            template_path = Path(directory) / "example.json"
            template_path.write_text(
                '{"message": "Hello, {name}."}',
                encoding="utf-8",
            )
            engine = TemplateEngine(directory)

            first_load = engine.load_template("example.json")
            template_path.write_text('{"message": "Changed."}', encoding="utf-8")
            second_load = engine.load_template("example.json")

            self.assertIs(first_load, second_load)
            self.assertEqual(
                engine.render(first_load["message"], name="Ada"),
                "Hello, Ada.",
            )

    def test_template_engine_rejects_non_string_template_values(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "invalid.json").write_text(
                '{"message": 42}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "JSON object of string values",
            ):
                TemplateEngine(directory).load_template("invalid.json")


if __name__ == "__main__":
    unittest.main()
