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
    "framework": "TensorFlow",
    "architecture": "Custom CNN",
    "task": "Classification",
    "num_classes": "Unknown",
    "optimizer": "Adam",
    "loss_function": "SparseCategoricalCrossentropy",
    "input_type": "Image"
},
   {
    "framework": "Keras",
    "architecture": "Custom CNN",
    "task": "Unknown",
    "num_classes": "Unknown",
    "optimizer": "Adam",
    "loss_function": "CategoricalCrossentropy",
    "input_type": "Image"
},
   {
    "framework": "Keras",
    "architecture": "ResNet-50",
    "task": "Unknown",
    "num_classes": "Unknown",
    "optimizer": "Adam",
    "loss_function": "CategoricalCrossentropy",
    "input_type": "Image"
},
    {
    "framework": "Scikit-Learn",
    "architecture": "RandomForestClassifier",
    "task": "Classification",
    "num_classes": "Unknown",
    "optimizer": "Unknown",
    "loss_function": "Unknown",
    "input_type": "Tabular"
},
]

for data in test_cases:
    parsed = normalize_fields(data)         
    model_info = build_model_info(parsed)
    result = select_xai(model_info)
    print(format_result(model_info, result))
    print("-" * 50)