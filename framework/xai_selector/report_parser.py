"""
Parses the Model Analyzer's JSON report into a structured dict.
Each field is normalized to {"value": ..., "confidence": ...} so the
rest of the pipeline (adapter, formatter) doesn't need to change.
"""

import json

DEFAULT_CONFIDENCE = 100.0

def normalize_fields(raw: dict) -> dict:
    """
    Converts a plain flat dict (e.g. {"architecture": "Custom CNN"})
    into the normalized shape the adapter expects
    (e.g. {"architecture": {"value": "Custom CNN", "confidence": 100.0}}).
    Also reusable directly by test scripts that fake raw analyzer output.
    """
    parsed = {}
    for field, value in raw.items():
        if isinstance(value, dict) and "value" in value:
            parsed[field.strip().lower()] = {
                "value": value["value"],
                "confidence": float(value.get("confidence", DEFAULT_CONFIDENCE))
            }
        else:
            parsed[field.strip().lower()] = {
                "value": value,
                "confidence": DEFAULT_CONFIDENCE
            }
    return parsed


def parse_model_report(filepath: str) -> dict:
    with open(filepath, "r") as f:
        raw = json.load(f)
    return normalize_fields(raw)