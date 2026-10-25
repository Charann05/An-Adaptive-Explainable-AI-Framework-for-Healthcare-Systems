import unittest
from pathlib import Path

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

MODEL_PATH = Path(__file__).with_name("test_rf_model.pkl")


class SHAPIntegrationTests(unittest.TestCase):
    def test_explains_random_forest_model(self):
        if not MODEL_PATH.is_file():
            self.skipTest(f"Model fixture not found: {MODEL_PATH}")

        try:
            import joblib
            import shap  # noqa: F401
        except ImportError as error:
            self.skipTest(f"SHAP dependencies are unavailable: {error}")

        from framework.xai_integration.engine import run_explanation

        model = joblib.load(MODEL_PATH)
        result = run_explanation(
            model=model,
            input_data=[[5.1, 3.5, 1.4, 0.2]],
            model_info={"category": "RandomForest", "input_type": "Tabular"},
            selector_result={
                "selected": "SHAP",
                "alternatives": ["FeatureImportance", "LIME"],
            },
        )
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()