import unittest
from pathlib import Path

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from adaptive_xai.explanation_generator.template_engine import TemplateEngine


class PackagedTemplateTests(unittest.TestCase):
    def test_renders_packaged_prediction_summary(self):
        template_directory = (
            Path(__file__).resolve().parents[1]
            / "adaptive_xai"
            / "explanation_generator"
            / "templates"
        )
        engine = TemplateEngine(template_directory)
        templates = engine.load_template("common.json")

        result = engine.render(
            templates["prediction_summary"],
            prediction="Diabetes",
        )

        self.assertEqual(result, "The model's answer is: Diabetes.")


if __name__ == "__main__":
    unittest.main()
