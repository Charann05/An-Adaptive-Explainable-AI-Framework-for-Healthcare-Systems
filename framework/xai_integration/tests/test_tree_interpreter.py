import joblib
import numpy as np

from ..engine import run_explanation


MODEL_PATH = "framework/xai_integration/tests/test_rf_model.pkl"


model = joblib.load(MODEL_PATH)


model_info = {
    "category": "RandomForest",
    "input_type": "Tabular"
}


input_data = np.array([
    [0.5, 0.2, 0.8, 0.3]
])


selector_result = {
    "selected": "TreeInterpreter",
    "alternatives": []
}


result = run_explanation(
    model=model,
    input_data=input_data,
    model_info=model_info,
    selector_result=selector_result
)


print(result)