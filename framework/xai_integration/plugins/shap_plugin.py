from ..base_plugin import XAIPlugin
from ..output_schema import ExplanationResult
from ..plugin_registry import register_plugin


@register_plugin("SHAP")
class SHAPPlugin(XAIPlugin):

    name = "SHAP"

    SUPPORTED_CATEGORIES = {
        "Tree",
        "DecisionTree",
        "RandomForest",
        "XGBoost",
    }

    def is_compatible(self, model_info: dict) -> bool:
        return (
            model_info.get("category")
            in self.SUPPORTED_CATEGORIES
        )

    def explain(self, model, input_data, model_info: dict, adapter=None) -> ExplanationResult:

        import numpy as np
        import shap

        explainer = shap.TreeExplainer(model)

        values = explainer.shap_values(
            np.asarray(input_data)
        )

        prediction = int(
            model.predict(input_data)[0]
        )

        feature_names = model_info.get(
            "feature_names"
        )

        if isinstance(values, list):
            class_values = values[prediction][0]
        else:
            values = np.asarray(values)

            if values.ndim == 3:
                class_values = values[0, :, prediction]
            else:
                class_values = values[0]

        ranking = []

        for index, score in enumerate(class_values):

            name = (
                feature_names[index]
                if feature_names and index < len(feature_names)
                else f"feature_{index}"
            )

            ranking.append({
                "feature": name,
                "importance": float(score),
            })

        ranking.sort(
            key=lambda item: abs(item["importance"]),
            reverse=True,
        )

        return ExplanationResult(
            technique_used=self.name,
            prediction=prediction,
            feature_ranking=ranking,
            raw_output=np.asarray(
                class_values
            ).tolist(),
        )