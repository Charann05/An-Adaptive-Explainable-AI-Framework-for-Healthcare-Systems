import unittest
from unittest.mock import patch

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from adaptive_xai.dependency_manager.manager import _prepare_packages


class DependencyManagerTests(unittest.TestCase):
    def test_records_already_installed_packages_without_installing(self):
        dependencies = [
            {"pip": "example-package", "import": "example_module"},
        ]

        with (
            patch(
                "adaptive_xai.dependency_manager.manager.is_installed",
                return_value=True,
            ),
            patch(
                "adaptive_xai.dependency_manager.manager.install_package"
            ) as install_package,
        ):
            result = _prepare_packages(dependencies)

        self.assertEqual(
            result,
            {"installed": ["example-package"], "failed": []},
        )
        install_package.assert_not_called()

    def test_reports_failed_install_without_retrying_or_external_install(self):
        dependencies = [
            {"pip": "example-package", "import": "example_module"},
        ]

        with (
            patch(
                "adaptive_xai.dependency_manager.manager.is_installed",
                return_value=False,
            ),
            patch(
                "adaptive_xai.dependency_manager.manager.install_package",
                return_value=False,
            ) as install_package,
        ):
            result = _prepare_packages(dependencies)

        self.assertEqual(
            result,
            {"installed": [], "failed": ["example-package"]},
        )
        install_package.assert_called_once_with("example-package")


if __name__ == "__main__":
    unittest.main()
