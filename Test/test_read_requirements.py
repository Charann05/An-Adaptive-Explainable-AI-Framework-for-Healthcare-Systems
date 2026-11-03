import tempfile
import unittest
from pathlib import Path

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from adaptive_xai.model_analyzer.read_requirements import scan_requirements


class ReadRequirementsTests(unittest.TestCase):
    def test_scans_utf16_requirements_file(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "requirements.txt").write_bytes(
                "scikit-learn==1.5.0\n".encode("utf-16")
            )
            evidences = {"framework": []}
            file_info = [{"name": "requirements.txt", "extension": ".txt"}]

            scan_requirements(project, evidences, file_info)

        self.assertEqual(
            evidences["framework"],
            [{"value": "Scikit-Learn", "source": "requirements", "weight": 6}],
        )

    def test_scans_utf8_requirements_file(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "requirements.txt").write_text(
                "scikit-learn==1.5.0\n",
                encoding="utf-8",
            )
            evidences = {"framework": []}
            file_info = [{"name": "requirements.txt", "extension": ".txt"}]

            scan_requirements(project, evidences, file_info)

        self.assertEqual(
            evidences["framework"],
            [{"value": "Scikit-Learn", "source": "requirements", "weight": 6}],
        )


if __name__ == "__main__":
    unittest.main()
