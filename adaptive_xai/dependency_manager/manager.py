from pathlib import Path

from .constants import MODEL_DEPENDENCIES, XAI_DEPENDENCIES
from .checker import is_installed
from .installer import install_package


def _prepare_packages(dependencies):
    installed = []
    failed = []

    for dependency in dependencies:
        pip_name = dependency["pip"]
        import_name = dependency["import"]

        if is_installed(import_name):
            print(f"{pip_name} is already installed.")
            installed.append(pip_name)
        elif install_package(pip_name):
            installed.append(pip_name)
        else:
            failed.append(pip_name)

    return {
        "installed": installed,
        "failed": failed,
    }


def prepare_dependencies(xai_technique):
    if xai_technique not in XAI_DEPENDENCIES:
        raise ValueError(f"No dependency information found for {xai_technique}")

    return _prepare_packages(XAI_DEPENDENCIES[xai_technique])


def prepare_model_dependencies(framework, model_path=None):
    framework_key = adaptive_xai.strip().lower()
    aliases = {
        "pytorch": "pytorch",
        "torch": "pytorch",
        "tensorflow": "tensorflow",
        "keras": "keras",
        "sklearn": "sklearn",
        "scikit-learn": "scikit-learn",
        "xgboost": "xgboost",
    }
    dependency_key = aliases.get(framework_key)
    if dependency_key is None:
        raise ValueError(
            f"No model dependency information found for {framework}."
        )

    dependencies = MODEL_DEPENDENCIES[dependency_key]
    if (
        dependency_key == "xgboost"
        and model_path is not None
        and Path(model_path).suffix.lower() in {".pkl", ".pickle", ".joblib"}
    ):
        dependencies = [
            *MODEL_DEPENDENCIES["xgboost"],
            {
                "pip": "joblib",
                "import": "joblib",
            }
        ]

    return _prepare_packages(dependencies)