import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from adaptive_xai import run
from adaptive_xai.model_analyzer.report_generator import generate_report


class FeatureMetadataPromptTests(unittest.TestCase):
    def test_prompts_for_count_and_exact_feature_names(self):
        evidence = {
            "num_inputs": None,
            "feature_names": None,
        }

        with patch("builtins.input", side_effect=["2", "age", "glucose"]):
            feature_count, feature_names = run.prompt_for_feature_metadata(
                evidence
            )

        self.assertEqual(feature_count, 2)
        self.assertEqual(feature_names, ["age", "glucose"])
        self.assertEqual(evidence["num_inputs"], {"value": 2})
        self.assertEqual(
            evidence["feature_names"],
            {"value": ["age", "glucose"]},
        )

    def test_reuses_saved_feature_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            report_path = project / "model_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "num_inputs": 2,
                        "feature_names": ["age", "glucose"],
                    }
                ),
                encoding="utf-8",
            )
            evidence = {
                "num_inputs": None,
                "feature_names": {"value": "Unknown"},
            }

            run.reuse_report_feature_metadata(project, evidence)

            self.assertEqual(evidence["num_inputs"], {"value": 2})
            self.assertEqual(
                evidence["feature_names"],
                {"value": ["age", "glucose"]},
            )

    def test_report_generation_preserves_manual_feature_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "model_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "num_inputs": 2,
                        "feature_names": ["age", "glucose"],
                        "framework": "scikit-learn",
                    }
                ),
                encoding="utf-8",
            )

            generate_report(
                {
                    "num_inputs": None,
                    "feature_names": None,
                    "framework": {"value": "scikit-learn"},
                },
                output_file=str(report_path),
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["num_inputs"], 2)
            self.assertEqual(report["feature_names"], ["age", "glucose"])
            self.assertEqual(report["framework"], "scikit-learn")

    def test_save_feature_metadata_updates_existing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            report_path = project / "model_report.json"
            report_path.write_text(
                json.dumps({"framework": "scikit-learn"}),
                encoding="utf-8",
            )

            run.save_feature_metadata(
                project,
                2,
                ["age", "glucose"],
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["feature_names"], ["age", "glucose"])
            self.assertEqual(report["num_inputs"], 2)
            self.assertEqual(report["framework"], "scikit-learn")


if __name__ == "__main__":
    unittest.main()
