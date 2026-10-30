import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from framework.model_analyzer import analyzer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AnalyzerCliTests(unittest.TestCase):
    def test_analyze_console_command_targets_analyzer_main(self):
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
            project_config = tomllib.load(file)

        self.assertEqual(
            project_config["project"]["scripts"]["analyze"],
            "framework.model_analyzer.analyzer:main",
        )

    def test_main_runs_model_analyzer_only(self):
        with patch.object(analyzer, "analyze_model") as analyze_model:
            analyzer.main()

        analyze_model.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
