import os
import torch

from .tiny_cnn import TinyCNN
from ..engine import run_explanation


model_path = os.path.join(
    os.path.dirname(__file__),
    "test_cnn_model.pt"
)

model = torch.load(
    model_path,
    weights_only=False
)

model_info = {
    "category": "CNN",
    "input_type": "Image"
}

selector_result = {
    "selected": "GradCAM",
    "alternatives": ["SHAP", "LIME"]
}

fake_image = torch.rand(1, 3, 32, 32)

result = run_explanation(
    model=model,
    input_data=fake_image,
    model_info=model_info,
    selector_result=selector_result
)

print(result)