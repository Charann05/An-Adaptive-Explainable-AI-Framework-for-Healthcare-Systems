import importlib.util
import io
import sys
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from adaptive_xai.xai_integration.model_loader import ModelLoader
from adaptive_xai.xai_integration.exceptions import (
    ModelCompatibilityError,
)
from adaptive_xai.cli_display import (
    print_model_compatibility_error,
    print_model_load_error,
)


@unittest.skipUnless(
    importlib.util.find_spec("sklearn")
    and importlib.util.find_spec("joblib"),
    "scikit-learn and joblib are required",
)
class SklearnModelLoaderCompatibilityTests(unittest.TestCase):
    def test_includes_saved_and_installed_sklearn_versions_on_failure(self):
        import joblib
        from sklearn.exceptions import InconsistentVersionWarning

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "incompatible-model.pkl"
            model_path.touch()

            def fail_loading(_path):
                warnings.warn(
                    InconsistentVersionWarning(
                        estimator_name="Pipeline",
                        current_sklearn_version="1.9.0",
                        original_sklearn_version="1.6.1",
                    )
                )
                raise AttributeError(
                    "module 'sklearn.compose._column_transformer' has no "
                    "attribute '_RemainderColsList'"
                )

            with (
                patch(
                    "adaptive_xai.xai_integration.model_loader."
                    "prepare_model_dependencies",
                    return_value={"failed": []},
                ),
                patch.object(joblib, "load", side_effect=fail_loading),
            ):
                with self.assertRaises(ModelCompatibilityError) as raised:
                    ModelLoader().load(model_path, "scikit-learn")

        self.assertEqual(raised.exception.package, "scikit-learn")
        self.assertEqual(raised.exception.saved_versions, ("1.6.1",))
        self.assertEqual(raised.exception.installed_version, "1.9.0")
        self.assertIsInstance(raised.exception.__cause__, AttributeError)

    def test_displays_version_pin_and_safe_recovery_steps(self):
        error = ModelCompatibilityError(
            "Version mismatch",
            package="scikit-learn",
            saved_versions=("1.6.1",),
            installed_version="1.9.0",
        )
        output = io.StringIO()

        with redirect_stdout(output):
            print_model_load_error(error)

        displayed = output.getvalue()
        self.assertIn("scikit-learn 1.6.1", displayed)
        self.assertIn("scikit-learn 1.9.0", displayed)
        self.assertIn("scikit-learn==1.6.1", displayed)
        self.assertIn("separate environment", displayed)
        self.assertIn("not a guarantee", displayed)
        self.assertIn("Your model file was not modified", displayed)

    def test_reports_incompatible_tree_pickle_format(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "incompatible-model.pkl"
            model_path.touch()

            with (
                patch(
                    "adaptive_xai.xai_integration.model_loader."
                    "prepare_model_dependencies",
                    return_value={"failed": []},
                ),
                patch.object(
                    ModelLoader,
                    "_load_joblib",
                    side_effect=ValueError(
                        "node array from the pickle has an incompatible dtype"
                    ),
                ),
            ):
                with self.assertRaises(ModelCompatibilityError) as raised:
                    ModelLoader().load(model_path, "scikit-learn")

        self.assertIn("saved with an incompatible version", str(raised.exception))
        self.assertIn("incompatible dtype", str(raised.exception.__cause__))

    def test_prints_highlighted_compatibility_guidance(self):
        output = io.StringIO()
        with redirect_stdout(output):
            print_model_compatibility_error(
                "The model uses an incompatible scikit-learn version."
            )

        self.assertIn("MODEL VERSION INCOMPATIBILITY", output.getvalue())
        self.assertIn("Retrain and save the model", output.getvalue())
        self.assertIn("\033[1;37;41m", output.getvalue())

    def test_loads_models_using_legacy_module_paths(self):
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        self._assert_legacy_module_loads(
            joblib,
            RandomForestClassifier,
            "sklearn.ensemble.forest",
        )
        self._assert_legacy_module_loads(
            joblib,
            DecisionTreeClassifier,
            "sklearn.tree.tree",
        )

    def _assert_legacy_module_loads(
        self,
        joblib,
        model_class,
        legacy_module_name,
    ):
        legacy_module = ModuleType(legacy_module_name)
        legacy_module.__dict__[model_class.__name__] = model_class
        model = model_class()
        original_module_name = model_class.__module__
        existing_legacy_module = sys.modules.get(legacy_module_name)

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "legacy-model.joblib"
            try:
                with patch.object(
                    model_class,
                    "__module__",
                    legacy_module_name,
                ), patch.dict(
                    sys.modules,
                    {legacy_module_name: legacy_module},
                ):
                    joblib.dump(model, model_path)

                with patch(
                    "adaptive_xai.xai_integration.model_loader."
                    "prepare_model_dependencies",
                    return_value={"failed": []},
                ):
                    loaded_model = ModelLoader().load(
                        model_path,
                        "scikit-learn",
                    )

                self.assertIsInstance(loaded_model, model_class)
            finally:
                model_class.__module__ = original_module_name
                if existing_legacy_module is None:
                    sys.modules.pop(legacy_module_name, None)
                else:
                    sys.modules[legacy_module_name] = existing_legacy_module


if __name__ == "__main__":
    unittest.main()
