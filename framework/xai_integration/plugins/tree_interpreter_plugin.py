import numpy as np

from ..base_plugin import XAIPlugin
from ..output_schema import ExplanationResult
from ..plugin_registry import register_plugin


@register_plugin("TreeInterpreter")
class TreeInterpreterPlugin(XAIPlugin):

    name = "TreeInterpreter"

    SUPPORTED_CATEGORIES = {
        "Tree",
        "DecisionTree",
        "RandomForest",
    }

    def is_compatible(self, model_info: dict) -> bool:
        return (
            model_info.get("category")
            in self.SUPPORTED_CATEGORIES
        )

    def explain(self, model, input_data, model_info: dict, adapter=None) -> ExplanationResult:

        from treeinterpreter import treeinterpreter as ti

        prediction, bias, contributions = ti.predict(
            model,
            np.asarray(input_data),
        )

        prediction_values = np.asarray(prediction[0])
        if prediction_values.ndim == 0 or prediction_values.size == 1:
            predicted_class = prediction_values.reshape(-1)[0].item()
            feature_contributions = np.asarray(contributions[0]).reshape(-1)
        else:
            predicted_class = int(np.argmax(prediction_values))
            feature_contributions = contributions[0][:, predicted_class]

        feature_names = model_info.get("feature_names")

        ranking = []

        for index, score in enumerate(feature_contributions):

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
            prediction=predicted_class,
            feature_ranking=ranking,
            raw_output={
                "bias": bias[0].tolist(),
                "contributions": feature_contributions.tolist(),
            },
        )