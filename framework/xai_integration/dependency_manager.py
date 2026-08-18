from dataclasses import dataclass
import importlib.util
import subprocess
import sys


@dataclass(frozen=True)
class Dependency:
    package: str
    module: str


@dataclass
class DependencyResult:
    technique: str
    available: bool
    installed: bool = False
    package: str | None = None
    error: str | None = None


DEPENDENCIES: dict[str, Dependency] = {
    "GradCAM": Dependency(
        package="grad-cam",
        module="pytorch_grad_cam",
    ),
    "SHAP": Dependency(
        package="shap",
        module="shap",
    ),
    "LIME": Dependency(
        package="lime",
        module="lime",
    ),
    "FeatureImportance": Dependency(
        package="scikit-learn",
        module="sklearn",
    ),
    "TreeInterpreter": Dependency(
        package="treeinterpreter",
        module="treeinterpreter",
    ),
}


class DependencyManager:

    def is_available(self, module: str) -> bool:
        """
        Check whether a Python module is currently available.
        """
        return importlib.util.find_spec(module) is not None

    def ensure(
        self,
        technique: str,
        auto_install: bool = True,
        timeout: int = 120,
    ) -> DependencyResult:

        dependency = DEPENDENCIES.get(technique)

        if dependency is None:
            return DependencyResult(
                technique=technique,
                available=False,
                error=(
                    f"No dependency configuration found "
                    f"for technique '{technique}'."
                ),
            )

        # Already installed
        if self.is_available(dependency.module):
            return DependencyResult(
                technique=technique,
                available=True,
                package=dependency.package,
            )

        # Installation disabled
        if not auto_install:
            return DependencyResult(
                technique=technique,
                available=False,
                package=dependency.package,
                error=(
                    f"Required package '{dependency.package}' "
                    "is not installed and automatic installation "
                    "is disabled."
                ),
            )

        # Attempt installation
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    dependency.package,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

        except subprocess.TimeoutExpired:
            return DependencyResult(
                technique=technique,
                available=False,
                package=dependency.package,
                error=(
                    f"Installation of '{dependency.package}' "
                    f"timed out after {timeout} seconds."
                ),
            )

        except subprocess.CalledProcessError as exc:
            return DependencyResult(
                technique=technique,
                available=False,
                package=dependency.package,
                error=(
                    exc.stderr.strip()
                    or f"pip failed with exit code {exc.returncode}"
                ),
            )

        except Exception as exc:
            return DependencyResult(
                technique=technique,
                available=False,
                package=dependency.package,
                error=str(exc),
            )

        # Verify installation
        if not self.is_available(dependency.module):
            return DependencyResult(
                technique=technique,
                available=False,
                package=dependency.package,
                error=(
                    f"Package '{dependency.package}' reported a "
                    "successful installation, but the module could "
                    "not be imported."
                ),
            )

        return DependencyResult(
            technique=technique,
            available=True,
            installed=True,
            package=dependency.package,
        )


dependency_manager = DependencyManager()


def prepare_dependencies(
    technique: str,
    auto_install: bool = True,
) -> DependencyResult:

    return dependency_manager.ensure(
        technique=technique,
        auto_install=auto_install,
    )