import sys
import tempfile
import unittest
import io
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import call, patch

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from adaptive_xai.dependency_manager.manager import prepare_model_dependencies
from adaptive_xai.xai_integration.exceptions import ModelLoadError
from adaptive_xai.xai_integration.model_loader import ModelLoader
from adaptive_xai.cli_display import print_model_load_error


class XGBoostPickleDependencyTests(unittest.TestCase):
    def test_displays_friendly_xgboost_interface_diagnosis(self):
        error = ModelLoadError(
            "Failed to load model using scikit-learn."
        )
        error.__cause__ = ModelLoadError(
            "The installed XGBoost package does not expose any supported "
            "scikit-learn estimator classes."
        )
        output = io.StringIO()

        with redirect_stdout(output):
            print_model_load_error(error)

        self.assertIn("XGBOOST MODEL INTERFACE UNAVAILABLE", output.getvalue())
        self.assertIn("native JSON or UBJ", output.getvalue())
        self.assertIn("scikit-learn estimator classes", output.getvalue())

    def test_displays_generic_model_load_diagnosis_and_cause(self):
        cause = ValueError("invalid model structure")
        error = ModelLoadError("Failed to load model.")
        error.__cause__ = cause
        output = io.StringIO()

        with redirect_stdout(output):
            print_model_load_error(error)

        self.assertIn("MODEL LOAD FAILED", output.getvalue())
        self.assertIn("Details: invalid model structure", output.getvalue())

    def test_explains_column_transformer_version_mismatch_calmly(self):
        cause = AttributeError(
            "module 'sklearn.compose._column_transformer' has no attribute "
            "'_RemainderColsList'"
        )
        error = ModelLoadError("Failed to load model.")
        error.__cause__ = cause
        output = io.StringIO()

        with redirect_stdout(output):
            print_model_load_error(error)

        displayed = output.getvalue()
        self.assertIn("SCIKIT-LEARN PIPELINE VERSION MISMATCH", displayed)
        self.assertIn("not a problem with adap_xai.py", displayed)
        self.assertIn("has not modified your model file", displayed)
        self.assertIn("retrain and re-save", displayed)
        self.assertIn("separate environment", displayed)
        self.assertIn("Details:", displayed)

    def test_aliases_missing_legacy_sklearn_module_to_public_estimators(self):
        xgboost = ModuleType("xgboost")
        xgboost.XGBClassifier = type("XGBClassifier", (), {})
        module_name = "xgboost.sklearn"
        previous_module = sys.modules.get(module_name)

        def import_module(name):
            if name == module_name:
                raise ModuleNotFoundError(
                    "No module named 'xgboost.sklearn'",
                    name=module_name,
                )
            if name == "xgboost":
                return xgboost
            raise AssertionError(f"Unexpected import requested: {name}")

        try:
            with patch(
                "adaptive_xai.xai_integration.model_loader.importlib."
                "import_module",
                side_effect=import_module,
            ):
                ModelLoader._ensure_legacy_xgboost_sklearn_module()

            compatibility_module = sys.modules[module_name]
            self.assertIs(
                compatibility_module.XGBClassifier,
                xgboost.XGBClassifier,
            )
            self.assertFalse(hasattr(compatibility_module, "XGBRegressor"))
            self.assertIs(xgboost.sklearn, compatibility_module)
        finally:
            if previous_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module

    def test_does_not_create_alias_without_public_xgboost_estimators(self):
        xgboost = ModuleType("xgboost")
        module_name = "xgboost.sklearn"
        previous_module = sys.modules.get(module_name)

        def import_module(name):
            if name == module_name:
                raise ModuleNotFoundError(
                    "No module named 'xgboost.sklearn'",
                    name=module_name,
                )
            if name == "xgboost":
                return xgboost
            raise AssertionError(f"Unexpected import requested: {name}")

        try:
            with patch(
                "adaptive_xai.xai_integration.model_loader.importlib."
                "import_module",
                side_effect=import_module,
            ):
                with self.assertRaisesRegex(
                    ModelLoadError,
                    "does not expose any supported",
                ):
                    ModelLoader._ensure_legacy_xgboost_sklearn_module()
        finally:
            if previous_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module

    def test_scikit_learn_pickle_retries_after_missing_xgboost_module(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pkl"
            model_path.touch()

            with (
                patch(
                    "adaptive_xai.xai_integration.model_loader."
                    "prepare_model_dependencies",
                    side_effect=[
                        {"failed": []},
                        {"failed": []},
                    ],
                ) as prepare_dependencies,
                patch.object(
                    ModelLoader,
                    "_load_joblib",
                    side_effect=[
                        ModuleNotFoundError(
                            "No module named 'xgboost.sklearn'",
                            name="xgboost.sklearn",
                        ),
                        "loaded XGBoost model",
                    ],
                ) as load_joblib,
            ):
                result = ModelLoader().load(model_path, "scikit-learn")

        self.assertEqual(result, "loaded XGBoost model")
        self.assertEqual(
            prepare_dependencies.call_args_list,
            [
                call("scikit-learn", model_path),
                call("xgboost", model_path),
            ],
        )
        self.assertEqual(load_joblib.call_count, 2)

    def test_reports_failed_xgboost_dependency_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pkl"
            model_path.touch()

            with (
                patch(
                    "adaptive_xai.xai_integration.model_loader."
                    "prepare_model_dependencies",
                    side_effect=[
                        {"failed": []},
                        {"failed": ["xgboost"]},
                    ],
                ),
                patch.object(
                    ModelLoader,
                    "_load_joblib",
                    side_effect=ModuleNotFoundError(
                        "No module named 'xgboost.sklearn'",
                        name="xgboost.sklearn",
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    ModelLoadError,
                    "requires XGBoost",
                ):
                    ModelLoader().load(model_path, "scikit-learn")

    def test_xgboost_pickle_dependency_set_includes_xgboost_and_joblib(self):
        with patch(
            "adaptive_xai.dependency_manager.manager._prepare_packages",
            return_value={"installed": [], "failed": []},
        ) as prepare_packages:
            prepare_model_dependencies("xgboost", "model.pkl")

        self.assertEqual(
            prepare_packages.call_args.args[0],
            [
                {"pip": "xgboost", "import": "xgboost"},
                {"pip": "joblib", "import": "joblib"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
