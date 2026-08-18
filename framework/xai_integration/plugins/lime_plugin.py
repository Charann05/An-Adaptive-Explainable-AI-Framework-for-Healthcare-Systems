from ..base_plugin import XAIPlugin
from ..output_schema import ExplanationResult
from ..plugin_registry import register_plugin


@register_plugin("LIME")
class LIMEPlugin(XAIPlugin):

    name = "LIME"

    def is_compatible(self, model_info: dict) -> bool:
        return (
            model_info.get("input_type") == "Tabular"
        )

    def explain(self, model, input_data, model_info: dict, adapter=None) -> ExplanationResult:

        import numpy as np
        from lime.lime_tabular import LimeTabularExplainer

        if not hasattr(model, "predict_proba"):
            raise ValueError(
                "LIME tabular classification requires "
                "model.predict_proba()."
            )

        training_data = model_info.get("background_data")

        if training_data is None:
            raise ValueError(
                "LIME requires background_data from the "
                "input/data layer."
            )

        training_data = np.asarray(training_data)
        instance = np.asarray(input_data)[0]

        feature_names = model_info.get(
            "feature_names"
        )

        explainer = LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            mode="classification",
            discretize_continuous=True,
        )

        explanation = explainer.explain_instance(
            instance,
            model.predict_proba,
            num_features=len(instance),
        )

        ranking = [
            {
                "feature": name,
                "importance": float(score),
            }
            for name, score in explanation.as_list()
        ]

        prediction = int(
            model.predict(np.asarray(input_data))[0]
        )

        return ExplanationResult(
            technique_used=self.name,
            prediction=prediction,
            feature_ranking=ranking,
            raw_output=explanation.as_list(),
        )