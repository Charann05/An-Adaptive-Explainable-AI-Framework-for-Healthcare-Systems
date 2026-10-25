import unittest
from pathlib import Path

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

MODEL_PATH = Path(__file__).with_name("test_rf_model.pkl")


class TreeInterpreterIntegrationTests(unittest.TestCase):
    def test_explains_random_forest_model(self):
        if not MODEL_PATH.is_file():
            self.skipTest(f"Model fixture not found: {MODEL_PATH}")

        try:
            import joblib
            import numpy as np
            import treeinterpreter  # noqa: F401
        except ImportError as error:
            self.skipTest(f"TreeInterpreter dependencies are unavailable: {error}")

        from framework.xai_integration.engine import run_explanation

        model = joblib.load(MODEL_PATH)
        result = run_explanation(
            model=model,
            input_data=np.array([[0.5, 0.2, 0.8, 0.3]]),
            model_info={"category": "RandomForest", "input_type": "Tabular"},
            selector_result={"selected": "TreeInterpreter", "alternatives": []},
        )
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()