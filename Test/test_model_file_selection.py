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
    def test_generated_runner_clears_heatmap_and_loads_tensorflow_images_nhwc(self):
        try:
            from PIL import Image
        except ImportError as error:
            self.skipTest(f"Image dependencies are unavailable: {error}")

        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            integration = PersistentIntegration(project)
            integration._write_entrypoint(
                IntegrationManifest(
                    model_path=str(project / "model.keras"),
                    model_info={
                        "framework": "tensorflow",
                        "input_type": "Image",
                    },
                    selector_result={"selected": "GradCAM"},
                    technique="GradCAM",
                )
            )
            tree = ast.parse(
                integration.entrypoint.read_text(encoding="utf-8")
            )

            old_heatmap = project / "gradcam_explanation.png"
            old_heatmap.write_bytes(b"old heatmap")
            startup_cleanup = next(
                node
                for node in tree.body
                if isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
            )
            visualization_path_assignment = next(
                node
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id == "VISUALIZATION_PATH"
                    for target in node.targets
                )
            )
            exec(
                compile(
                    ast.Module(
                        body=[
                            visualization_path_assignment,
                            startup_cleanup,
                        ],
                        type_ignores=[],
                    ),
                    str(integration.entrypoint),
                    "exec",
                ),
                {
                    "__name__": "__main__",
                    "__file__": str(integration.entrypoint),
                    "Path": Path,
                },
            )
            self.assertFalse(old_heatmap.exists())

            image_loader = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_load_image"
            )
            namespace = {
                "INTEGRATION": {
                    "model_info": {"framework": "tensorflow"},
                },
            }
            exec(
                compile(
                    ast.Module(body=[image_loader], type_ignores=[]),
                    str(integration.entrypoint),
                    "exec",
                ),
                namespace,
            )
            image_path = project / "input.png"
            Image.new("RGB", (4, 4), (255, 0, 0)).save(image_path)
            loaded_image = namespace["_load_image"](str(image_path))

            self.assertEqual(loaded_image.shape, (1, 224, 224, 3))
            self.assertEqual(loaded_image[0, 0, 0].tolist(), [1.0, 0.0, 0.0])

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
                'VISUALIZATION_PATH = Path(__file__).with_name("gradcam_explanation.png")',
                generated_source,
            )
            self.assertIn(
                "VISUALIZATION_PATH.unlink(missing_ok=True)",
                generated_source,
            )
            self.assertLess(
                generated_source.index("VISUALIZATION_PATH.unlink(missing_ok=True)"),
                generated_source.index("MODEL_PATH = Path(INTEGRATION"),
            )
            self.assertIn(
                "        MODEL_PATH,",
                generated_source,
            )
            self.assertIn("BACKGROUND_DATA =", generated_source)
            self.assertIn("format_xai_output", generated_source)
            self.assertIn(
                "print(format_xai_output(output, input_type, visualization_path))",
                generated_source,
            )
            self.assertIn(
                "print(format_explanation(output, input_type, visualization_path))",
                generated_source,
            )
            self.assertLess(
                generated_source.index(
                    "print(format_xai_output(output, input_type, visualization_path))"
                ),
                generated_source.index(
                    "print(format_explanation(output, input_type, visualization_path))"
                ),
            )
            self.assertIn(
                'framework in {"tensorflow", "keras"}',
                generated_source,
            )
            self.assertIn("return array[np.newaxis, ...]", generated_source)
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
