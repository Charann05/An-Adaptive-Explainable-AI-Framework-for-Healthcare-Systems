"""
Parses the Model Analyzer's JSON report.

The Model Analyzer already produces a structured JSON file, so this
module simply loads the JSON and normalizes the field names
(lowercase + trimmed) for consistency across the adaptive_xai.
"""

import json


def normalize_fields(raw: dict) -> dict:
    """
    Normalizes JSON keys by trimming whitespace and converting
    them to lowercase.
    """
    return {
        field.strip().lower(): value
        for field, value in raw.items()
    }


def parse_model_report(filepath: str) -> dict:
    """
    Reads the Model Analyzer JSON report and returns a normalized dictionary.
    """
    with open(filepath, "r") as file:
        raw = json.load(file)

    return normalize_fields(raw)