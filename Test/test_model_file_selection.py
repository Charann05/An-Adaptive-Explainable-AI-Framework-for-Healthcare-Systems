import ast
import tempfile
import unittest
from pathlib import Path

import pandas as pd

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from framework.xai_integration.persistent import (
    IntegrationManifest,
    PersistentIntegration,
    find_model_file,
    find_model_files,
)


class ModelFileSelectionTests(unittest.TestCase):
    def test_lists_matching_files_in_existing_preference_order(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "other.pkl").touch()
            (project / "z_model.joblib").touch()
            (project / "a_model.pkl").touch()
            (project / "model.h5").touch()

            model_files = find_model_files(project, "scikit-learn")

            self.assertEqual(
                [path.name for path in model_files],
                ["a_model.pkl", "z_model.joblib", "other.pkl"],
            )
            self.assertEqual(
                find_model_file(project, "scikit-learn"),
                model_files[0],
            )

    def test_generated_entrypoint_exposes_editable_model_path(self):
        with tempfile.TemporaryDirectory() as directory:
            integration = PersistentIntegration(directory)
            manifest = IntegrationManifest(
                model_path=str(Path(directory) / "model.pkl"),
                model_info={"framework": "scikit-learn"},
                selector_result={"selected": "SHAP"},
                technique="SHAP",
            )

            integration._write_entrypoint(manifest)
            generated_source = integration.entrypoint.read_text(
                encoding="utf-8"
            )
            ast.parse(generated_source)

            self.assertIn(
                'MODEL_PATH = Path(INTEGRATION["model_path"])',
                generated_source,
            )
            self.assertIn(
                "        MODEL_PATH,",
                generated_source,
            )
            self.assertIn("BACKGROUND_DATA =", generated_source)
            self.assertIn("def preprocess_input(input_data: Any):", generated_source)
            self.assertIn("def _apply_preprocessing(input_data: Any):", generated_source)
            self.assertIn(
                'model_info["background_data"] = _apply_preprocessing(BACKGROUND_DATA)',
                generated_source,
            )

    def test_tabular_background_uses_matching_columns_in_model_order(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            pd.DataFrame(
                {
                    "label": [0, 1, 0],
                    "glucose": [95, 140, 110],
                    "age": [30, 50, 40],
                }
            ).to_csv(project / "training_data.csv", index=False)
            integration = PersistentIntegration(project)

            model_info = integration._resolve_background_data(
                {
                    "input_type": "Tabular",
                    "feature_names": ["age", "glucose"],
                },
                {"selected": "SHAP", "alternatives": ["LIME"]},
            )

            self.assertEqual(len(model_info["background_data"]), 3)
            self.assertCountEqual(
                model_info["background_data"],
                [[30, 95], [50, 140], [40, 110]],
            )

    def test_no_matching_tabular_dataset_leaves_background_unset(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            pd.DataFrame({"unrelated": [1, 2]}).to_csv(
                project / "data.csv",
                index=False,
            )
            integration = PersistentIntegration(project)

            model_info = integration._resolve_background_data(
                {
                    "input_type": "Tabular",
                    "feature_names": ["age", "glucose"],
                },
                {"selected": "LIME", "alternatives": []},
            )

            self.assertNotIn("background_data", model_info)


if __name__ == "__main__":
    unittest.main()
