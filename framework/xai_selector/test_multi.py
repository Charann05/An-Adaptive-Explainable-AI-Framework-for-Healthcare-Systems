from .report_parser import normalize_fields
from .adapter import build_model_info
from .selector import select_xai
from .report_formatter import format_result

test_cases = [
    {
        "framework": "TensorFlow",
        "architecture": "Custom CNN",
        "task": "Classification",
        "num_classes": "Unknown",
        "optimizer": "Adam",
        "loss_function": "BinaryCrossentropy",
        "input_type": "Image"
    },
    {
        "framework": "sklearn",
        "architecture": "RandomForestClassifier",
        "task": "Classification",
        "input_type": "Tabular"
    },
    {
        "framework": "sklearn",
        "architecture": "DecisionTreeClassifier",
        "task": "Classification",
        "input_type": "Tabular"
    },
    {
        "framework": "xgboost",
        "architecture": "XGBClassifier",
        "task": "Classification",
        "input_type": "Tabular"
    },
    {
        "framework": "sklearn",
        "architecture": "LogisticRegression",
        "task": "Classification",
        "input_type": "Tabular"
    },
    {
        "framework": "unknown",
        "architecture": "MysteryModel",
        "task": "Unknown",
        "input_type": "Tabular"
    },
]

for data in test_cases:
    parsed = normalize_fields(data)         
    model_info = build_model_info(parsed)
    result = select_xai(model_info)
    print(format_result(model_info, result))
    print("-" * 50)