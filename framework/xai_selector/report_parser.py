"""
Parses the Model Analyzer's .txt report into a structured dict.
Each field is stored with its value AND confidence score.
"""

import re

LINE_PATTERN = re.compile(r"^(\w+):\s*(.+?)\s*\((\d+(?:\.\d+)?)%\)$")

def parse_model_report(filepath: str) -> dict:
    parsed = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            match = LINE_PATTERN.match(line)
            if match:
                field, value, confidence = match.groups()
                parsed[field.strip().lower()] = {
                    "value": value.strip(),
                    "confidence": float(confidence)
                }
    return parsed