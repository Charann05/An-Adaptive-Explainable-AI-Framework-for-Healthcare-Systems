from .constants import XAI_DEPENDENCIES
from .checker import is_installed
from .installer import install_package

def prepare_dependencies(xai_technique):
    if xai_technique not in XAI_DEPENDENCIES:
        raise ValueError(f"No dependency information found for {xai_technique}")

    installed = []
    failed = []

    for dependency in XAI_DEPENDENCIES[xai_technique]:
        pip_name = dependency["pip"]
        import_name = dependency["import"]

        if is_installed(import_name):
            print(f"{pip_name} is already installed.")
            installed.append(pip_name)

        else:
            success = install_package(pip_name)
            if success:
                installed.append(pip_name)
            else:
                failed.append(pip_name)

    return {
        "installed": installed,
        "failed": failed
    }