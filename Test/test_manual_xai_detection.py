import tempfile
import unittest
from pathlib import Path

import nbformat

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from adaptive_xai.xai_integration.manual_detection import detect_manual_xai_usage


class ManualXaiDetectionTests(unittest.TestCase):
    def test_detects_xai_imports_in_source_and_notebooks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "explain.py").write_text(
                "from shap import Explainer\n"
                "import pytorch_grad_cam\n",
                encoding="utf-8",
            )
            (root / "notes.py").write_text(
                "# import lime\n"
                "print('treeinterpreter')\n",
                encoding="utf-8",
            )
            notebook = nbformat.v4.new_notebook(
                cells=[
                    nbformat.v4.new_code_cell(
                        "from lime.lime_tabular import LimeTabularExplainer"
                    )
                ]
            )
            nbformat.write(notebook, root / "analysis.ipynb")
            (root / "adap_xai.py").write_text(
                "import shap\n",
                encoding="utf-8",
            )

            detected = detect_manual_xai_usage(root)

        self.assertEqual(
            detected,
            {
                "GradCAM": ["explain.py"],
                "LIME": ["analysis.ipynb"],
                "SHAP": ["explain.py"],
            },
        )

    def test_returns_empty_when_no_supported_xai_imports_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.py").write_text(
                "import sklearn\n",
                encoding="utf-8",
            )

            self.assertEqual(detect_manual_xai_usage(root), {})


if __name__ == "__main__":
    unittest.main()
