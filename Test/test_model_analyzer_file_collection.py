import tempfile
import unittest
from pathlib import Path

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from framework.model_analyzer.load_model_files import (
    collect_files,
    is_valid_project_folder,
)
from framework.model_analyzer.classify_file_type import get_file_info
from framework.model_analyzer.read_input_type import detect_input_type


class ModelAnalyzerFileCollectionTests(unittest.TestCase):
    def test_excludes_generated_integration_from_project_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "adap_xai.py").write_text(
                "import cv2\n"
                "image = cv2.imread('input.png')\n",
                encoding="utf-8",
            )
            (project / "model.py").write_text(
                "from sklearn.tree import DecisionTreeClassifier\n",
                encoding="utf-8",
            )

            collected_files = collect_files(project)
            evidences = {"input_type": []}
            detect_input_type(
                project,
                evidences,
                get_file_info(project, collected_files),
            )

            self.assertNotIn("adap_xai.py", collected_files)
            self.assertIn("model.py", collected_files)
            self.assertNotIn(
                "Image",
                [item["value"] for item in evidences["input_type"]],
            )

    def test_excludes_exact_framework_report_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "model_report.json").write_text("{}", encoding="utf-8")
            (project / "config.json").write_text("{}", encoding="utf-8")
            nested = project / "reports"
            nested.mkdir()
            (nested / "model_report.json").write_text("{}", encoding="utf-8")

            collected_files = collect_files(project)

            self.assertNotIn("model_report.json", collected_files)
            self.assertNotIn("reports/model_report.json", collected_files)
            self.assertIn("config.json", collected_files)

    def test_report_file_alone_does_not_make_folder_a_project(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "model_report.json").write_text("{}", encoding="utf-8")

            self.assertFalse(is_valid_project_folder(project))


if __name__ == "__main__":
    unittest.main()
