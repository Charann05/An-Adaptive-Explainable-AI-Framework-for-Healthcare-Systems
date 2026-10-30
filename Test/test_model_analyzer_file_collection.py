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


class ModelAnalyzerFileCollectionTests(unittest.TestCase):
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
