from ..base_plugin import XAIPlugin
from ..output_schema import ExplanationResult
from ..plugin_registry import register_plugin


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

        if not hasattr(model, "feature_importances_"):
            raise ValueError(
                "Model does not expose feature_importances_."
            )

        importances = model.feature_importances_

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

        prediction = None

        if input_data is not None:
            prediction = model.predict(input_data)[0]

        return ExplanationResult(
            technique_used=self.name,
            prediction=prediction,
            feature_ranking=ranking,
            raw_output=importances.tolist(),
        )