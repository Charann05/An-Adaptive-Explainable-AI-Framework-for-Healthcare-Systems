import json
import os


def _is_unknown(value):
    return value is None or (
        isinstance(value, str)
        and value.strip().lower() in {"", "unknown", "none"}
    )


def generate_report(final_evidence, output_file="model_report.json"):
    report = {}

    for category, evidence in final_evidence.items():
        if evidence is None:
            report[category] = "Unknown"
        else:
            report[category] = evidence["value"]

    if os.path.isfile(output_file):
        with open(output_file, "r", encoding="utf-8") as file:
            previous_report = json.load(file)
        if isinstance(previous_report, dict):
            for key in ("num_inputs", "feature_names"):
                if _is_unknown(report.get(key)) and not _is_unknown(
                    previous_report.get(key)
                ):
                    report[key] = previous_report[key]

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    # print(f"Report generated: {output_file}")
    print("Report saved at:", os.path.abspath(output_file))