"""
Converts a parsed Model Analyzer report into the model_info dict
that select_xai_technique() expects, and carries confidence along.
"""

from .mapping_rules import normalize_category

CONFIDENCE_THRESHOLD = 60.0

def build_model_info(parsed_report: dict) -> dict:
    architecture = parsed_report.get("architecture", {}).get("value")
    framework = parsed_report.get("framework", {}).get("value", "Unknown")
    input_type = parsed_report.get("input_type", {}).get("value", "Unknown")
    num_classes = parsed_report.get("num_classes", {}).get("value")
    task = parsed_report.get("task", {}).get("value")   # NEW

    category = normalize_category(raw_category=None, raw_architecture=architecture)

    confidence = parsed_report.get("architecture", {}).get("confidence", 100.0)

    return {
        "framework": framework,
        "architecture": architecture,
        "category": category,
        "input_type": input_type,
        "num_classes": num_classes,
        "task": task,                                    # NEW
        "confidence": confidence,
        "low_confidence": confidence < CONFIDENCE_THRESHOLD
    }