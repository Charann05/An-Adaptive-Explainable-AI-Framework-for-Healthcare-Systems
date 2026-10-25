import unittest
from pathlib import Path

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

MODEL_PATH = Path(__file__).with_name("test_rf_model.pkl")


class LIMEIntegrationTests(unittest.TestCase):
    def test_explains_random_forest_model(self):
        if not MODEL_PATH.is_file():
            self.skipTest(f"Model fixture not found: {MODEL_PATH}")

        try:
            import joblib
            import lime  # noqa: F401
        except ImportError as error:
            self.skipTest(f"LIME dependencies are unavailable: {error}")

        from framework.xai_integration.engine import run_explanation

        model = joblib.load(MODEL_PATH)
        result = run_explanation(
            model=model,
            input_data=[[5.1, 3.5, 1.4, 0.2]],
            model_info={
                "category": "RandomForest",
                "input_type": "Tabular",
                "background_data": [
                    [5.1, 3.5, 1.4, 0.2],
                    [6.2, 3.4, 5.4, 2.3],
                    [5.9, 3.0, 5.1, 1.8],
                ],
                "framework": "sklearn",
            },
            selector_result={"selected": "LIME", "alternatives": []},
        )
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()