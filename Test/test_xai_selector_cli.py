import json
import tempfile
import tomllib
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from adaptive_xai.xai_selector import cli


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class XaiSelectorCliTests(unittest.TestCase):
    def test_console_command_is_registered(self):
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
            scripts = tomllib.load(file)["project"]["scripts"]

        self.assertEqual(
            scripts["xai-select"],
            "adaptive_xai.xai_selector.cli:main",
        )

    def test_continues_setup_from_report_without_running_analyzer(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "edited-report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "framework": "scikit-learn",
                        "architecture": "RandomForestClassifier",
                        "input_type": "Tabular",
                        "task": "Classification",
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch("sys.argv", ["xai-select", str(report_path)]),
                patch.object(cli, "run_project") as run_project,
            ):
                cli.main()

        run_project.assert_called_once()
        self.assertEqual(
            run_project.call_args.kwargs["project_root"],
            report_path.parent,
        )
        self.assertEqual(
            run_project.call_args.kwargs["report_path"],
            report_path,
        )
        self.assertEqual(
            run_project.call_args.kwargs["final_evidence"]["architecture"],
            {"value": "RandomForestClassifier"},
        )

    def test_reports_missing_required_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "model_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "framework": "scikit-learn",
                        "architecture": "Unknown",
                        "input_type": "Tabular",
                    }
                ),
                encoding="utf-8",
            )
            stderr = StringIO()

            with (
                patch("sys.argv", ["xai-select", str(report_path)]),
                redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as error:
                    cli.main()

        self.assertEqual(error.exception.code, 2)
        self.assertIn("architecture", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
