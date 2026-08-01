from .adapter import build_model_info
from .selector import select_xai
from .report_formatter import format_result

# Simulate different Model Analyzer outputs directly (skipping the txt parsing step)
sample_reports = [
    {"framework": {"value": "PyTorch", "confidence": 100.0},
     "architecture": {"value": "EfficientNet-B0", "confidence": 70.0},
     "input_type": {"value": "Image", "confidence": 70.0}},

    {"framework": {"value": "sklearn", "confidence": 100.0},
     "architecture": {"value": "RandomForestClassifier", "confidence": 90.0},
     "input_type": {"value": "Tabular", "confidence": 90.0}},

    {"framework": {"value": "sklearn", "confidence": 100.0},
     "architecture": {"value": "DecisionTreeClassifier", "confidence": 85.0},
     "input_type": {"value": "Tabular", "confidence": 85.0}},

    {"framework": {"value": "xgboost", "confidence": 100.0},
     "architecture": {"value": "XGBClassifier", "confidence": 88.0},
     "input_type": {"value": "Tabular", "confidence": 88.0}},

    {"framework": {"value": "sklearn", "confidence": 100.0},
     "architecture": {"value": "LogisticRegression", "confidence": 80.0},
     "input_type": {"value": "Tabular", "confidence": 80.0}},

    {"framework": {"value": "unknown", "confidence": 30.0},
     "architecture": {"value": "MysteryModel", "confidence": 30.0},
     "input_type": {"value": "Tabular", "confidence": 30.0}},
]

for parsed in sample_reports:
    model_info = build_model_info(parsed)
    result = select_xai(model_info)
    print(format_result(model_info, result))
    print("-" * 50)