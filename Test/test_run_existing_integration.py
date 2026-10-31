import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from adaptive_xai import run


class ExistingIntegrationStartupTests(unittest.TestCase):
    def test_setup_prompts_and_persists_missing_tabular_feature_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            model_path = project / "model.pkl"
            model_path.touch()
            report_path = project / "model_report.json"
            report_path.write_text(
                '{"framework": "scikit-learn"}',
                encoding="utf-8",
            )
            integration = type(
                "Integration",
                (),
                {
                    "entrypoint": project / "adap_xai.py",
                    "exists": lambda self: False,
                    "create": lambda self, **kwargs: setattr(
                        self,
                        "manifest",
                        kwargs,
                    ),
                },
            )()
            evidence = {
                "framework": {"value": "scikit-learn"},
                "architecture": {"value": "DecisionTree"},
                "task": {"value": "Classification"},
                "input_type": {"value": "Tabular"},
                "feature_names": None,
                "num_inputs": None,
            }
            selector_result = {"selected": "SHAP", "alternatives": []}

            with (
                patch.object(run, "get_model_folder", return_value=project),
                patch.object(run, "PersistentIntegration", return_value=integration),
                patch.object(run, "detect_manual_xai_usage", return_value={}),
                patch.object(run, "analyze_model", return_value=evidence),
                patch.object(run, "is_valid_analysis", return_value=True),
                patch.object(run, "select_xai", return_value=selector_result),
                patch.object(run, "prepare_dependencies"),
                patch.object(run, "find_model_files", return_value=[model_path]),
                patch.object(run, "build_model_info", return_value={}),
                patch("builtins.input", side_effect=["2", "age", "glucose"]),
                redirect_stdout(StringIO()),
            ):
                run.main()

            saved_report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_report["num_inputs"], 2)
            self.assertEqual(
                saved_report["feature_names"],
                ["age", "glucose"],
            )
            self.assertEqual(
                integration.manifest["model_info"]["feature_names"],
                ["age", "glucose"],
            )
            self.assertEqual(
                integration.manifest["model_info"]["num_inputs"],
                2,
            )

    def test_analysis_validation_rejects_unknown_evidence(self):
        self.assertFalse(
            run.is_valid_analysis(
                {
                    "framework": {"value": "Unknown"},
                    "architecture": {"value": "DecisionTree"},
                }
            )
        )
        self.assertFalse(
            run.is_valid_analysis(
                {
                    "framework": {"value": "scikit-learn"},
                    "architecture": None,
                }
            )
        )

    def test_invalid_analysis_prints_highlighted_stop_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            output = StringIO()

            with (
                patch.object(run, "get_model_folder", return_value=project),
                patch.object(run, "detect_manual_xai_usage", return_value={}),
                patch.object(
                    run,
                    "analyze_model",
                    return_value={
                        "framework": {"value": "scikit-learn"},
                        "architecture": None,
                    },
                ),
                redirect_stdout(output),
            ):
                run.main()

            message = output.getvalue()
            self.assertIn(
                "The selected folder may not be the correct model project.",
                message,
            )
            self.assertEqual(
                message.count(
                    "The selected folder may not be the correct model project."
                ),
                1,
            )
            self.assertIn("Setup stopped because", message)
            self.assertIn("could not identify the model architecture/type", message)
            self.assertIn("rerun the setup", message)

    def test_completed_setup_prints_run_command_and_preprocessing_guidance(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            model_path = project / "model.pkl"
            model_path.touch()
            integration = type(
                "Integration",
                (),
                {
                    "entrypoint": project / "adap_xai.py",
                    "exists": lambda self: False,
                    "create": lambda self, **kwargs: None,
                },
            )()
            evidence = {
                "framework": {"value": "scikit-learn"},
                "architecture": {"value": "DecisionTree"},
                "task": {"value": "Classification"},
                "input_type": {"value": "Tabular"},
                "feature_names": {"value": ["feature"]},
            }
            output = StringIO()

            with (
                patch.object(run, "get_model_folder", return_value=project),
                patch.object(run, "PersistentIntegration", return_value=integration),
                patch.object(run, "detect_manual_xai_usage", return_value={}),
                patch.object(run, "analyze_model", return_value=evidence),
                patch.object(run, "is_valid_analysis", return_value=True),
                patch.object(
                    run,
                    "select_xai",
                    return_value={"selected": "SHAP", "alternatives": []},
                ),
                patch.object(run, "prepare_dependencies"),
                patch.object(run, "find_model_files", return_value=[model_path]),
                patch.object(run, "build_model_info", return_value={}),
                redirect_stdout(output),
            ):
                run.main()

            self.assertIn("preprocess_input()", output.getvalue())
            self.assertIn("python adap_xai.py", output.getvalue())

    def test_declining_replacement_preserves_existing_integration(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            entrypoint = project / "adap_xai.py"
            original_contents = "# existing user integration\n"
            entrypoint.write_text(original_contents, encoding="utf-8")
            output = StringIO()

            with (
                patch.object(run, "get_model_folder", return_value=project),
                patch.object(run, "analyze_model") as analyze_model,
                patch("builtins.input", return_value="no"),
                redirect_stdout(output),
            ):
                run.main()

            analyze_model.assert_not_called()
            self.assertEqual(
                entrypoint.read_text(encoding="utf-8"),
                original_contents,
            )
            self.assertIn("XAI integration already exists", output.getvalue())
            self.assertIn("manual edits", output.getvalue())
            self.assertIn("MODEL_PATH", output.getvalue())
            self.assertIn(
                "model file itself will not be changed",
                output.getvalue(),
            )
            self.assertIn(
                "Keeping the existing integration unchanged",
                output.getvalue(),
            )

    def test_confirming_replacement_continues_to_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "adap_xai.py").write_text(
                "# existing user integration\n",
                encoding="utf-8",
            )

            with (
                patch.object(run, "get_model_folder", return_value=project),
                patch("builtins.input", return_value="yes"),
                patch.object(run, "analyze_model", return_value={}) as analyze_model,
                patch.object(run, "is_valid_analysis", return_value=False),
                redirect_stdout(StringIO()),
            ):
                run.main()

        analyze_model.assert_called_once()

    def test_confirmation_reprompts_on_unrecognized_answer(self):
        with (
            patch("builtins.input", side_effect=["maybe", "n"]) as prompt,
            redirect_stdout(StringIO()),
        ):
            result = run.confirm_integration_replacement("adap_xai.py")

        self.assertFalse(result)
        self.assertEqual(prompt.call_count, 2)

    def test_manual_integration_confirmation_can_decline(self):
        detected_usage = {
            "SHAP": ["explain.py"],
            "LIME": ["notebooks/analysis.ipynb"],
        }
        output = StringIO()

        with (
            patch("builtins.input", return_value="no"),
            redirect_stdout(output),
        ):
            result = run.confirm_manual_integration_replacement(
                detected_usage
            )

        self.assertFalse(result)
        self.assertIn("Possible manual XAI integration", output.getvalue())
        self.assertIn("SHAP: explain.py", output.getvalue())
        self.assertIn("not edit your existing source files", output.getvalue())
        self.assertIn("may duplicate explanations", output.getvalue())

    def test_detected_manual_integration_can_stop_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "explain.py").write_text(
                "import shap\n",
                encoding="utf-8",
            )
            output = StringIO()

            with (
                patch.object(run, "get_model_folder", return_value=project),
                patch.object(run, "analyze_model") as analyze_model,
                patch("builtins.input", return_value="no"),
                redirect_stdout(output),
            ):
                run.main()

            analyze_model.assert_not_called()
            self.assertIn("Possible manual XAI integration", output.getvalue())
            self.assertIn("SHAP: explain.py", output.getvalue())
            self.assertIn(
                "Keeping the existing project code unchanged",
                output.getvalue(),
            )


if __name__ == "__main__":
    unittest.main()
