import os

from .report_parser import parse_model_report
from .adapter import build_model_info
from .selector import select_xai
from .report_formatter import format_result

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(CURRENT_DIR, "model_report.txt")

parsed = parse_model_report(REPORT_PATH)
model_info = build_model_info(parsed)
result = select_xai(model_info)

print("=== Parsed Report ===")
print(parsed)

print("\n=== Model Info ===")
print(model_info)

print("\n=== Selector Result ===")
print(result)
print(format_result(model_info, result))