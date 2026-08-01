"""
Converts a parsed Model Analyzer report into the model_info dict
that select_xai_technique() expects, and carries confidence along.
"""

from .mapping_rules import normalize_category

CONFIDENCE_THRESHOLD = 60.0   # below this, we flag low confidence

def build_model_info(parsed_report: dict) -> dict:
    architecture = parsed_report.get("architecture", {}).get("value")
    framework = parsed_report.get("framework", {}).get("value", "Unknown")
    input_type = parsed_report.get("input_type", {}).get("value", "Unknown")
    num_classes = parsed_report.get("num_classes", {}).get("value")

    category = normalize_category(raw_category=None, raw_architecture=architecture)

    # Use architecture's confidence as the confidence in the detected category
    confidence = parsed_report.get("architecture", {}).get("confidence", 100.0)

    return {
        "framework": framework,
        "architecture": architecture,
        "category": category,
        "input_type": input_type,
        "num_classes": num_classes,
        "confidence": confidence,
        "low_confidence": confidence < CONFIDENCE_THRESHOLD
    }