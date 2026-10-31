"""
Converts the parsed Model Analyzer report into the standardized
model_info dictionary expected by the XAI Selector.
"""

from .mapping_rules import normalize_category

def build_model_info(parsed_report: dict) -> dict:
    architecture = parsed_report.get("architecture")
    framework = parsed_report.get("framework", "Unknown")
    input_type = parsed_report.get("input_type", "Unknown")
    num_classes = parsed_report.get("num_classes")
    task = parsed_report.get("task")  

    category = normalize_category(
        raw_category=None, 
        raw_architecture=architecture
    )

    return {
        "framework": framework,
        "architecture": architecture,
        "category": category,
        "input_type": input_type,
        "num_classes": num_classes,
        "task": task,                               
    }