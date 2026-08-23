from .model_analyzer.analyzer import analyze_model
from .xai_selector.selector import select_xai
from .xai_selector.report_formatter import format_result
from .dependency_manager import prepare_dependencies
from .cli_display import print_banner, print_status

def get_value(evidence):
    if evidence is None:
        return "Unknown"

    return evidence["value"]


def main():

    print_banner()
    print_status("Initializing the framework")

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

    # Display selector result FIRST
    print(format_result(model_info, result))

    # Get selected XAI technique
    selected_xai = result["selected"]

    # Prepare XAI dependencies
    prepare_dependencies(selected_xai)


if __name__ == "__main__":
    main()