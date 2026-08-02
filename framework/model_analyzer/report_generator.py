import json
import os

def generate_report(final_evidence, output_file="model_report.json"):
    report = {}

    for category, evidence in final_evidence.items():
        if evidence is None:
            report[category] = "Unknown"
        else:
            report[category] = evidence["value"]

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"Report generated: {output_file}")
    print("Report saved at:", os.path.abspath(output_file))