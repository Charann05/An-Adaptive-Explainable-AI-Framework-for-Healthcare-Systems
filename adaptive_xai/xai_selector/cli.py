import argparse
import json
from pathlib import Path

from adaptive_xai.run import run_project


REQUIRED_REPORT_FIELDS = ("framework", "architecture", "input_type")


def _validate_report(report, report_path):
    if not isinstance(report, dict):
        raise ValueError(f"Expected a JSON object in {report_path}.")

    missing = [
        field
        for field in REQUIRED_REPORT_FIELDS
        if not isinstance(report.get(field), str)
        or report[field].strip().lower() in {"", "unknown", "none"}
    ]
    if missing:
        fields = ", ".join(missing)
        raise ValueError(
            f"{report_path} is missing usable values for: {fields}. "
            "Update those fields and run xai-select again."
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Continue XAI setup using an existing model_report.json, "
            "skipping only model analysis."
        )
    )
    parser.add_argument(
        "report",
        nargs="?",
        default="model_report.json",
        help="Path to model_report.json (default: ./model_report.json)",
    )
    args = parser.parse_args()
    report_path = Path(args.report).expanduser().resolve()

    if not report_path.is_file():
        parser.error(
            f"Report file not found: {report_path}. Run 'analyze' first "
            "or provide a valid model_report.json path."
        )

    try:
        with report_path.open("r", encoding="utf-8") as file:
            report = json.load(file)
        _validate_report(report, report_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))

    evidence = {
        key: {"value": value}
        for key, value in report.items()
    }
    run_project(
        project_root=report_path.parent,
        final_evidence=evidence,
        report_path=report_path,
    )
