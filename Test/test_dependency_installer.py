import subprocess
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from adaptive_xai.dependency_manager.installer import install_package


class DependencyInstallerTests(unittest.TestCase):
    def test_timed_out_install_reports_error_and_stops_spinner(self):
        output = StringIO()
        timeout_error = subprocess.TimeoutExpired(
            cmd="pip install example-package",
            timeout=2,
            stderr="Temporary failure resolving package index",
        )

        with (
            patch(
                "adaptive_xai.dependency_manager.installer.subprocess.run",
                side_effect=timeout_error,
            ) as run,
            patch(
                "adaptive_xai.dependency_manager.installer.show_spinner"
            ) as show_spinner,
            redirect_stdout(output),
        ):
            result = install_package("example-package", timeout=2)

        self.assertFalse(result)
        run.assert_called_once()
        self.assertEqual(run.call_args.kwargs["timeout"], 2)
        show_spinner.assert_called_once()
        self.assertIn("timed out after 2 seconds", output.getvalue())
        self.assertIn("Temporary failure resolving package index", output.getvalue())

    def test_pip_network_failure_is_reported(self):
        output = StringIO()
        failed_install = subprocess.CompletedProcess(
            args=["pip", "install", "example-package"],
            returncode=1,
            stderr="Could not connect to package index",
        )

        with (
            patch(
                "adaptive_xai.dependency_manager.installer.subprocess.run",
                return_value=failed_install,
            ),
            patch(
                "adaptive_xai.dependency_manager.installer.show_spinner"
            ),
            redirect_stdout(output),
        ):
            result = install_package("example-package", timeout=2)

        self.assertFalse(result)
        self.assertIn("Check your network connection", output.getvalue())
        self.assertIn("Could not connect to package index", output.getvalue())

    def test_missing_pip_executable_is_reported(self):
        output = StringIO()

        with (
            patch(
                "adaptive_xai.dependency_manager.installer.subprocess.run",
                side_effect=FileNotFoundError("python executable not found"),
            ),
            patch(
                "adaptive_xai.dependency_manager.installer.show_spinner"
            ),
            redirect_stdout(output),
        ):
            result = install_package("example-package", timeout=2)

        self.assertFalse(result)
        self.assertIn("Could not start pip", output.getvalue())
        self.assertIn("python executable not found", output.getvalue())


if __name__ == "__main__":
    unittest.main()
