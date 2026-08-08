from framework.model_analyzer.analyzer import analyze_model
from framework.xai_selector.selector import select_xai
from framework.xai_selector.report_formatter import format_result


def get_value(evidence):
    if evidence is None:
        return "Unknown"

    return evidence["value"]


def main():
    # Run analyzer
    final_evidence = analyze_model()

    # Convert analyzer output to selector input
    model_info = {
        "framework": get_value(final_evidence.get("framework")),
        "architecture": get_value(final_evidence.get("architecture")),
        "category": get_value(final_evidence.get("task")),
        "input_type": get_value(final_evidence.get("input_type"))
    }

    # Run selector
    result = select_xai(model_info)

    # Display selector result
    print(format_result(model_info, result))


if __name__ == "__main__":
    main()