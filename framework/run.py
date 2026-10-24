import os
from pathlib import Path

from .model_analyzer.analyzer import analyze_model
from .model_analyzer.load_model_files import get_model_folder
from .xai_integration import PersistentIntegration, find_model_file
from .xai_selector.adapter import build_model_info
from .xai_selector.selector import select_xai
from .xai_selector.report_formatter import format_result
from .dependency_manager import prepare_dependencies
from .cli_display import print_banner, print_status, print_error


def get_value(evidence):
    if evidence is None:
        return "Unknown"

    return evidence["value"]


def is_valid_analysis(evidence):
    framework = evidence.get("framework", [])
    architecture = evidence.get("architecture", [])

    return bool(framework) and bool(architecture)


def main():
    print_banner()
    print_status("Initializing the framework")

    project_root = get_model_folder()
    original_directory = Path.cwd()
    try:
        os.chdir(project_root)

        final_evidence = analyze_model()

        if not is_valid_analysis(final_evidence):
            print_error("The selected folder may not be the correct model project.")
            return

        model_info = {
            "framework": get_value(final_evidence.get("framework")),
            "architecture": get_value(final_evidence.get("architecture")),
            "category": get_value(final_evidence.get("task")),
            "input_type": get_value(final_evidence.get("input_type")),
        }

        result = select_xai(model_info)
        print(format_result(model_info, result))

        selected_xai = result.get("selected")
        if not selected_xai:
            print_error("No suitable XAI technique was selected.")
            return

        prepare_dependencies(selected_xai)

        try:
            model_path = find_model_file(
                project_root,
                model_info["framework"],
            )
        except FileNotFoundError as error:
            print_error(str(error))
            return

        integration_model_info = build_model_info(model_info)
        feature_names = get_value(final_evidence.get("feature_names"))
        if feature_names != "Unknown":
            integration_model_info["feature_names"] = feature_names

        integration = PersistentIntegration(project_root)
        integration.create(
            model_path=model_path,
            model_info=integration_model_info,
            selector_result=result,
        )
        print_status(
            f"XAI integration created at {integration.entrypoint}"
        )
    finally:
        os.chdir(original_directory)


if __name__ == "__main__":
    main()