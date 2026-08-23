import os
import joblib

from ..engine import run_explanation

model_path = os.path.join(os.path.dirname(__file__), "test_rf_model.pkl")
model = joblib.load(model_path)

model_info = {"category": "RandomForest", "input_type": "Tabular"}
selector_result = {"selected": "FeatureImportance", "alternatives": ["SHAP", "LIME"]}

sample_input = [[5.1, 3.5, 1.4, 0.2]]  # one iris flower's measurements

result = run_explanation(model=model, input_data=sample_input,
                          model_info=model_info, selector_result=selector_result)
print(result)