import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVALUATION_DIR = Path(__file__).resolve().parent
TEST_RESULTS = EVALUATION_DIR / "results" / "test_results.csv"
ACCURACY_GRAPH = EVALUATION_DIR / "results" / "accuracy.png"
STACKED_GRAPH = EVALUATION_DIR / "results" / "accuracy_stacked.png"
LOLLIPOP_GRAPH = EVALUATION_DIR / "results" / "accuracy_lollipop.png"
RADAR_GRAPH = EVALUATION_DIR / "results" / "accuracy_radar.png"
DONUT_GRAPH = EVALUATION_DIR / "results" / "accuracy_donut.png"

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


def save_accuracy_graph(results, overall):
    accuracies = [
        (field.replace("_", " ").title(), calculate_accuracy(results, field))
        for field in EVALUATED_FIELDS
    ]
    accuracies = [item for item in accuracies if item[1] is not None]

    if overall is not None:
        accuracies.append(("Overall", overall))

    if not accuracies:
        print("No accuracy values available to plot.")
        return

    labels, values = zip(*accuracies)
    colors = ["#397b70" if label != "Overall" else "#cf704d" for label in labels]

    figure, axis = plt.subplots(figsize=(10, max(4, len(labels) * 0.45)))
    bars = axis.barh(labels, values, color=colors)
    axis.invert_yaxis()
    axis.set_xlim(0, 112)
    axis.set_xlabel("Accuracy (%)")
    axis.set_title("Model Analyzer Accuracy")
    axis.xaxis.grid(True, linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    axis.bar_label(bars, fmt="%.1f%%", padding=4)
    figure.tight_layout()
    figure.savefig(ACCURACY_GRAPH, dpi=160)
    plt.close(figure)
    print(f"Accuracy graph saved to: {ACCURACY_GRAPH}")


def save_alternative_charts(results, overall):
    field_labels = [field.replace("_", " ").title() for field in EVALUATED_FIELDS]
    field_accuracies = [calculate_accuracy(results, field) for field in EVALUATED_FIELDS]
    evaluable = [
        (label, accuracy)
        for label, accuracy in zip(field_labels, field_accuracies)
        if accuracy is not None
    ]

    project_count = len(results)
    pass_counts = [
        sum(row.get(field) == "PASS" for row in results)
        for field in EVALUATED_FIELDS
    ]
    fail_counts = [
        sum(row.get(field) == "FAIL" for row in results)
        for field in EVALUATED_FIELDS
    ]
    unevaluated_counts = [
        project_count - passed - failed
        for passed, failed in zip(pass_counts, fail_counts)
    ]

    figure, axis = plt.subplots(figsize=(11, max(5, len(field_labels) * 0.45)))
    left = [0.0] * len(field_labels)
    outcomes = [
        ("Pass", pass_counts, "#397b70"),
        ("Fail", fail_counts, "#cf704d"),
        ("Not evaluated", unevaluated_counts, "#c8cecc"),
    ]

    for label, counts, color in outcomes:
        widths = [count / project_count * 100 for count in counts]
        axis.barh(field_labels, widths, left=left, label=label, color=color)
        for index, (start, width) in enumerate(zip(left, widths)):
            if width >= 8:
                axis.text(
                    start + width / 2,
                    index,
                    f"{width:.0f}%",
                    ha="center",
                    va="center",
                    color="white" if label != "Not evaluated" else "#26312e",
                    fontsize=8,
                )
        left = [start + width for start, width in zip(left, widths)]

    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("Share of benchmark projects (%)")
    axis.set_title("Evaluation Outcomes by Field")
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
        ncol=3,
    )
    figure.tight_layout()
    figure.subplots_adjust(bottom=0.2)
    figure.savefig(STACKED_GRAPH, dpi=160)
    plt.close(figure)
    print(f"Stacked chart saved to: {STACKED_GRAPH}")

    if evaluable:
        labels, values = zip(*evaluable)
        if overall is not None:
            labels = (*labels, "Overall")
            values = (*values, overall)

        figure, axis = plt.subplots(figsize=(10, max(4, len(labels) * 0.45)))
        for index, (label, value) in enumerate(zip(labels, values)):
            color = "#cf704d" if label == "Overall" else "#397b70"
            axis.hlines(index, 0, value, color="#c8cecc", linewidth=2)
            axis.scatter(value, index, color=color, s=65, zorder=3)
            axis.text(value + 1.5, index, f"{value:.1f}%", va="center")

        axis.set_yticks(range(len(labels)), labels)
        axis.invert_yaxis()
        axis.set_xlim(0, 112)
        axis.set_xlabel("Accuracy (%)")
        axis.set_title("Model Analyzer Accuracy")
        axis.xaxis.grid(True, linestyle="--", alpha=0.35)
        axis.set_axisbelow(True)
        figure.tight_layout()
        figure.savefig(LOLLIPOP_GRAPH, dpi=160)
        plt.close(figure)
        print(f"Lollipop chart saved to: {LOLLIPOP_GRAPH}")

        radar_labels, radar_values = zip(*evaluable)
        angles = [
            index / len(radar_labels) * 2 * math.pi
            for index in range(len(radar_labels))
        ]
        closed_angles = (*angles, angles[0])
        closed_values = (*radar_values, radar_values[0])

        figure, axis = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
        axis.plot(closed_angles, closed_values, color="#397b70", linewidth=2)
        axis.fill(closed_angles, closed_values, color="#397b70", alpha=0.18)
        axis.set_xticks(angles, radar_labels)
        axis.set_ylim(0, 100)
        axis.set_yticks([25, 50, 75, 100])
        axis.set_title("Accuracy Profile by Field", pad=24)
        figure.tight_layout()
        figure.savefig(RADAR_GRAPH, dpi=160)
        plt.close(figure)
        print(f"Radar chart saved to: {RADAR_GRAPH}")

    pass_total = sum(pass_counts)
    fail_total = sum(fail_counts)
    evaluable_total = pass_total + fail_total

    if evaluable_total:
        figure, axis = plt.subplots(figsize=(7, 7))
        axis.pie(
            [pass_total, fail_total],
            labels=[f"Pass ({pass_total})", f"Fail ({fail_total})"],
            colors=["#397b70", "#cf704d"],
            startangle=90,
            counterclock=False,
            autopct="%1.1f%%",
            wedgeprops={"width": 0.38, "edgecolor": "white"},
            pctdistance=0.8,
        )
        axis.text(
            0,
            0,
            f"{pass_total / evaluable_total * 100:.1f}%\nOverall",
            ha="center",
            va="center",
            fontsize=16,
        )
        axis.set_title("Overall Pass / Fail")
        figure.tight_layout()
        figure.savefig(DONUT_GRAPH, dpi=160)
        plt.close(figure)
        print(f"Donut chart saved to: {DONUT_GRAPH}")


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
    save_accuracy_graph(results, overall)
    save_alternative_charts(results, overall)


if __name__ == "__main__":
    main()