import os
import json
from pathlib import Path

from .model_analyzer.analyzer import analyze_model
from .model_analyzer.load_model_files import get_model_folder
from .xai_integration import (
    PersistentIntegration,
    find_model_files,
)
from .xai_integration.manual_detection import detect_manual_xai_usage
from .xai_selector.adapter import build_model_info
from .xai_selector.selector import select_xai
from .xai_selector.report_formatter import format_result
from .dependency_manager import prepare_dependencies
from .cli_display import (
    print_banner,
    print_analysis_validation_error,
    print_error,
    print_model_load_error,
    print_step,
    print_status,
)
from .xai_integration.exceptions import ModelLoadError


def get_value(evidence):
    if evidence is None:
        return "Unknown"

    return evidence["value"]


def is_valid_analysis(evidence):
    if not isinstance(evidence, dict):
        return False

    def has_value(item):
        if isinstance(item, dict):
            item = item.get("value")
        if item is None:
            return False
        if isinstance(item, str):
            return item.strip().lower() not in {"", "unknown", "none"}
        return bool(item)

    return has_value(evidence.get("framework")) and has_value(
        evidence.get("architecture")
    )


def _evidence_value(evidence):
    if isinstance(evidence, dict):
        return evidence.get("value")
    return evidence


def _has_known_value(value):
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "unknown", "none"}
    return bool(value)


def reuse_report_feature_metadata(project_root, evidence, report_path=None):
    report_path = Path(report_path or Path(project_root) / "model_report.json")
    if not report_path.is_file():
        return evidence

    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)
    if not isinstance(report, dict):
        raise ValueError(f"Invalid analyzer report format: {report_path}")

    for key in ("num_inputs", "feature_names"):
        current_value = _evidence_value(evidence.get(key))
        saved_value = report.get(key)
        if not _has_known_value(current_value) and _has_known_value(saved_value):
            evidence[key] = {"value": saved_value}
    return evidence


def prompt_for_feature_metadata(evidence):
    feature_names = _evidence_value(evidence.get("feature_names"))
    feature_count = _evidence_value(evidence.get("num_inputs"))

    if _has_known_value(feature_count):
        try:
            feature_count = int(feature_count)
        except (TypeError, ValueError):
            feature_count = None
        if feature_count is not None and feature_count <= 0:
            feature_count = None

    valid_names = (
        isinstance(feature_names, list)
        and feature_names
        and all(isinstance(name, str) and name.strip() for name in feature_names)
    )
    if valid_names and not _has_known_value(feature_count):
        feature_count = len(feature_names)
        evidence["num_inputs"] = {"value": feature_count}

    if not _has_known_value(feature_count):
        while True:
            answer = input(
                "The number of input features could not be detected. "
                "Enter the number of input features: "
            ).strip()
            try:
                feature_count = int(answer)
            except ValueError:
                print_error("Enter a positive whole number.")
                continue
            if feature_count > 0:
                break
            print_error("The number of input features must be greater than zero.")

    if (
        not valid_names
        or len(feature_names) != feature_count
    ):
        print_step(
            f"Enter the {feature_count} feature names in the exact order "
            "expected by the model."
        )
        while True:
            entered_names = [
                input(f"Feature {index}: ").strip()
                for index in range(1, feature_count + 1)
            ]
            if all(entered_names) and len(set(entered_names)) == len(entered_names):
                feature_names = entered_names
                break
            print_error("Feature names must be non-empty and unique. Please retry.")

    evidence["num_inputs"] = {"value": feature_count}
    evidence["feature_names"] = {"value": feature_names}
    return feature_count, feature_names


def save_feature_metadata(
    project_root,
    feature_count,
    feature_names,
    report_path=None,
):
    report_path = Path(report_path or Path(project_root) / "model_report.json")
    if report_path.is_file():
        with report_path.open("r", encoding="utf-8") as file:
            report = json.load(file)
    else:
        report = {}
    if not isinstance(report, dict):
        raise ValueError(f"Invalid analyzer report format: {report_path}")

    report["num_inputs"] = feature_count
    report["feature_names"] = feature_names
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)


def prompt_for_model_file(project_root, framework):
    project_root = Path(project_root).expanduser().resolve()
    while True:
        entered_path = input(
            f"Enter the path to the {framework} model file "
            "(absolute or relative to the project folder, or press Enter to cancel): "
        ).strip().strip("\"'")
        if not entered_path:
            return None

        model_path = Path(entered_path).expanduser()
        if not model_path.is_absolute():
            model_path = project_root / model_path
        model_path = model_path.resolve()
        if model_path.is_file():
            return model_path

        print_error(f"Model file does not exist or is not a file: {model_path}")


def confirm_integration_replacement(entrypoint):
    print_status(f"An XAI integration already exists at {entrypoint}.")
    print_step(
        "Continuing will regenerate and overwrite adap_xai.py. Any manual "
        "edits, including your MODEL_PATH setting, will be lost."
    )
    print_step(
        "The model file itself will not be changed. The new integration may "
        "use a different detected model file or XAI technique."
    )
    while True:
        answer = input("Do you want to generate the integration again? [y/N]: ")
        normalized_answer = answer.strip().lower()
        if normalized_answer in {"y", "yes"}:
            return True
        if normalized_answer in {"", "n", "no"}:
            return False
        print_error("Please answer yes or no.")


def confirm_manual_integration_replacement(detected_usage):
    print_status(
        "Possible manual XAI integration detected in project source:"
    )
    for technique, source_files in detected_usage.items():
        print_step(f"{technique}: {', '.join(source_files)}")
    print_step(
        "This check is based on XAI library imports; it cannot confirm how "
        "the code is used."
    )
    print_step(
        "Continuing will create a new xai integration based on the xai technique that was selected."
    )
    print_step(
        "This will not edit your existing "
        "source files. Running both integrations may duplicate explanations "
        "or conflict with your current workflow."
    )

    while True:
        answer = input(
            "Do you still want to create the framework integration? [y/N]: "
        )
        normalized_answer = answer.strip().lower()
        if normalized_answer in {"y", "yes"}:
            return True
        if normalized_answer in {"", "n", "no"}:
            return False
        print_error("Please answer yes or no.")


def run_project(project_root, final_evidence=None, report_path=None):
    project_root = Path(project_root).expanduser().resolve()
    report_path = Path(report_path or project_root / "model_report.json")
    integration = PersistentIntegration(project_root)
    if integration.exists() and not confirm_integration_replacement(
        integration.entrypoint
    ):
        print_step("Keeping the existing integration unchanged.")
        return
    if not integration.exists():
        manual_xai_usage = detect_manual_xai_usage(project_root)
        if manual_xai_usage and not confirm_manual_integration_replacement(
            manual_xai_usage
        ):
            print_step("Keeping the existing project code unchanged.")
            return

    original_directory = Path.cwd()
    try:
        os.chdir(project_root)

        if final_evidence is None:
            final_evidence = analyze_model()
        final_evidence = reuse_report_feature_metadata(
            project_root,
            final_evidence,
            report_path,
        )

        if not is_valid_analysis(final_evidence):
            missing_details = []
            if not isinstance(final_evidence, dict):
                missing_details.append(
                    "The analyzer did not return a usable analysis report."
                )
            else:
                for key, label in (
                    ("framework", "machine-learning framework"),
                    ("architecture", "model architecture/type"),
                ):
                    evidence = final_evidence.get(key)
                    value = (
                        evidence.get("value")
                        if isinstance(evidence, dict)
                        else evidence
                    )
                    if value is None or (
                        isinstance(value, str)
                        and value.strip().lower() in {"", "unknown", "none"}
                    ) or not value:
                        missing_details.append(
                            f"The analyzer could not identify the {label}."
                        )

            print_analysis_validation_error(missing_details)
            return

        model_info = {
            "framework": get_value(final_evidence.get("framework")),
            "architecture": get_value(final_evidence.get("architecture")),
            "category": get_value(final_evidence.get("task")),
            "input_type": get_value(final_evidence.get("input_type")),
        }

        normalized_input_type = str(model_info["input_type"]).strip().lower()
        feature_names = _evidence_value(final_evidence.get("feature_names"))
        feature_count = _evidence_value(final_evidence.get("num_inputs"))
        if (
            normalized_input_type in {"", "unknown", "tabular"}
            and (
                not _has_known_value(feature_count)
                or not isinstance(feature_names, list)
                or not feature_names
            )
        ):
            feature_count, feature_names = prompt_for_feature_metadata(
                final_evidence
            )
            save_feature_metadata(
                project_root,
                feature_count,
                feature_names,
                report_path,
            )
            if normalized_input_type in {"", "unknown"}:
                model_info["input_type"] = "Tabular"

        result = select_xai(model_info)
        print(format_result(model_info, result))

        selected_xai = result.get("selected")
        if not selected_xai:
            print_error("No suitable XAI technique was selected.")
            return

        prepare_dependencies(selected_xai)

        model_files = find_model_files(
            project_root,
            model_info["framework"],
        )
        if model_files:
            project_root = Path(project_root).resolve()
            print_status("Compatible model files found:")
            for index, candidate in enumerate(model_files, start=1):
                print(f"  {index}. {candidate.relative_to(project_root)}")
            model_path = model_files[0]
            print_status(
                f"Selected model for XAI integration: "
                f"{model_path.relative_to(project_root)}"
            )
        else:
            print_error(
                f"Could not find a {model_info['framework']} model file "
                f"in {project_root}."
            )
            model_path = prompt_for_model_file(
                project_root,
                model_info["framework"],
            )
            if model_path is None:
                print_error("Model integration cancelled; no model file was selected.")
                return

        integration_model_info = build_model_info(model_info)
        feature_names = _evidence_value(final_evidence.get("feature_names"))
        if _has_known_value(feature_names):
            integration_model_info["feature_names"] = feature_names
        feature_count = _evidence_value(final_evidence.get("num_inputs"))
        if _has_known_value(feature_count):
            integration_model_info["num_inputs"] = feature_count

        integration = PersistentIntegration(project_root)
        try:
            integration.create(
                model_path=model_path,
                model_info=integration_model_info,
                selector_result=result,
            )
        except ModelLoadError as error:
            print_model_load_error(error)
            return
        print_status(
            f"XAI integration created at {integration.entrypoint}"
        )
        print_step(
            "To change the model file later, edit MODEL_PATH in adap_xai.py."
        )
        print_step(
            "If your model requires preprocessing, add the same steps used "
            "during training to preprocess_input() in adap_xai.py."
        )
        print_step(
            "Run the integrated project with: python adap_xai.py"
        )
    finally:
        os.chdir(original_directory)


def main():
    print_banner()
    print_status("Initializing the framework")
    run_project(get_model_folder())


if __name__ == "__main__":
    main()