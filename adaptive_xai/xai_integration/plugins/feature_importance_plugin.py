from ..base_plugin import XAIPlugin
from ..output_schema import ExplanationResult
from ..plugin_registry import register_plugin
from ..prediction_utils import predicted_value


@register_plugin("FeatureImportance")
class FeatureImportancePlugin(XAIPlugin):

    name = "FeatureImportance"

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

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = model.coef_
        else:
            raise ValueError(
                "Model does not expose feature_importances_ or coef_."
            )

        import numpy as np

        importances = np.asarray(importances)
        if importances.ndim > 1:
            importances = np.mean(np.abs(importances), axis=0)

        feature_names = model_info.get("feature_names")

        ranking = []

        for index, importance in enumerate(importances):

            name = (
                feature_names[index]
                if feature_names and index < len(feature_names)
                else f"feature_{index}"
            )

            ranking.append({
                "feature": name,
                "importance": float(importance),
            })

        ranking.sort(
            key=lambda item: abs(item["importance"]),
            reverse=True,
        )

        prediction = (
            None
            if input_data is None
            else predicted_value(model.predict(input_data), model_info.get("task"))
        )

        return ExplanationResult(
            technique_used=self.name,
            prediction=prediction,
            feature_ranking=ranking,
            raw_output=importances.tolist(),
            metadata={"explanation_scope": "global_model"},
        )