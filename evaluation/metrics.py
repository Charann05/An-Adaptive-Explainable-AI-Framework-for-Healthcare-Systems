import csv
from pathlib import Path

EVALUATION_DIR = Path(__file__).resolve().parent
TEST_RESULTS = EVALUATION_DIR / "results" / "test_results.csv"

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


def load_results():
    with open(TEST_RESULTS, "r", newline="") as file:
        return list(csv.DictReader(file))


def calculate_accuracy(results, field):
    evaluated = [
        row[field]
        for row in results
        if row.get(field) in ("PASS", "FAIL")
    ]

    if not evaluated:
        return None

    correct = evaluated.count("PASS")
    return correct / len(evaluated) * 100


def calculate_overall_accuracy(results):
    values = [
        row[field]
        for row in results
        for field in EVALUATED_FIELDS
        if row.get(field) in ("PASS", "FAIL")
    ]

    if not values:
        return None

    correct = values.count("PASS")
    return correct / len(values) * 100


def main():
    results = load_results()

    if not results:
        print("No evaluation results found.")
        return

    print("\nMODEL ANALYZER EVALUATION METRICS")
    print("=" * 55)

    for field in EVALUATED_FIELDS:
        accuracy = calculate_accuracy(results, field)

        if accuracy is None:
            print(f"{field:20}: N/A (no evaluable results)")
        else:
            print(f"{field:20}: {accuracy:.2f}%")

    overall = calculate_overall_accuracy(results)

    print("-" * 55)

    if overall is None:
        print("Overall accuracy     : N/A")
    else:
        print(f"Overall accuracy     : {overall:.2f}%")

    print("=" * 55)


if __name__ == "__main__":
    main()