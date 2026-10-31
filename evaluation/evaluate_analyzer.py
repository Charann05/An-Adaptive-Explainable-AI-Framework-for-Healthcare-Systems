import csv
import json
import os
from pathlib import Path

from adaptive_xai.model_analyzer.analyzer import analyze_model

EVALUATION_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = EVALUATION_DIR / "benchmark_projects"
EXPECTED_RESULTS = EVALUATION_DIR / "expected_results.json"
TEST_RESULTS = EVALUATION_DIR / "results/test_results.csv"

EVALUATED_FIELDS = [
    "framework",
    "architecture",
    "task",
    "input_type",
    "optimizer",
    "loss_function",
    "num_classes",
    "num_inputs",
    "dataset_columns",
    "feature_names",
    "target_column",
]


def load_expected_results():
    with open(EXPECTED_RESULTS, "r") as file:
        return json.load(file)


def analyze_project(project_path):
    original_directory = Path.cwd()

    try:
        os.chdir(project_path)
        return analyze_model()

    finally:
        os.chdir(original_directory)


def get_value(evidence):
    if evidence is None:
        return "Unknown"

    return evidence.get("value", "Unknown")


def compare_value(actual, expected):
    if expected is None:
        return "CANNOT_BE_EVALUATED"

    if actual == expected:
        return "PASS"

    return "FAIL"


def evaluate_project(benchmark):
    project_name = benchmark["project"]
    project_path = BENCHMARK_ROOT / project_name

    print(f"\nAnalyzing: {project_name}")

    if not project_path.exists():
        print(f"Project not found: {project_path}")
        return None

    try:
        actual = analyze_project(project_path)

    except Exception as error:
        print(f"Analyzer failed: {error}")
        return None

    result = {
        "project": project_name
    }

    for field in EVALUATED_FIELDS:
        actual_value = get_value(actual.get(field))
        expected_value = benchmark.get(field)

        result[field] = compare_value(
            actual_value,
            expected_value
        )

    return result


def save_results(results):
    fieldnames = [
        "project",
        *EVALUATED_FIELDS
    ]

    existing_results = {}

    if TEST_RESULTS.exists():
        with open(TEST_RESULTS, "r", newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                existing_results[row["project"]] = row

    for result in results:
        existing_results[result["project"]] = result

    with open(TEST_RESULTS, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(existing_results.values())


def print_result(result):
    print("\nEvaluation Result")
    print("-" * 50)

    for field in ["project", *EVALUATED_FIELDS]:
        print(f"{field:20}: {result[field]}")

    print("-" * 50)


def main():
    expected_results = load_expected_results()

    results = []

    for benchmark in expected_results:
        result = evaluate_project(benchmark)

        if result is None:
            continue

        results.append(result)
        print_result(result)

    save_results(results)

    print("\nTest results saved to:")
    print(TEST_RESULTS)


if __name__ == "__main__":
    main()