import os, joblib
from ..engine import run_explanation

model_path = os.path.join(os.path.dirname(__file__), "test_rf_model.pkl")
model = joblib.load(model_path)

model_info = {
    "category": "RandomForest", "input_type": "Tabular",
    "background_data": [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3], [5.9, 3.0, 5.1, 1.8]],
    "framework": "sklearn",
}
selector_result = {"selected": "LIME", "alternatives": []}
sample_input = [[5.1, 3.5, 1.4, 0.2]]

result = run_explanation(model=model, input_data=sample_input, model_info=model_info, selector_result=selector_result)
print(result)