from ..base_plugin import XAIPlugin
from ..output_schema import ExplanationResult
from ..plugin_registry import register_plugin
from ..adapters import resolve_adapter


@register_plugin("SHAP")
class SHAPPlugin(XAIPlugin):

    name = "SHAP"

    TREE_CATEGORIES = {"Tree", "DecisionTree", "RandomForest", "XGBoost"}

    def is_compatible(self, model_info: dict) -> bool:
        return True  # broadly compatible, per your knowledge base

    def explain(self, model, input_data, model_info: dict, adapter=None) -> ExplanationResult:
        import numpy as np
        import shap

        if adapter is None:
            adapter = resolve_adapter(model, model_info)

        category = model_info.get("category")
        feature_names = model_info.get("feature_names")
        input_array = np.asarray(input_data)

        if category in self.TREE_CATEGORIES:
            explainer = shap.TreeExplainer(model)
            raw_values = explainer.shap_values(input_array)

        elif adapter.framework == "pytorch":
            import torch
            background = torch.as_tensor(input_array).float()
            explainer = shap.DeepExplainer(model, background)
            raw_values = explainer.shap_values(background)

        elif adapter.framework == "tensorflow":
            explainer = shap.GradientExplainer(model, input_array)
            raw_values = explainer.shap_values(input_array)

        else:
            background = model_info.get("background_data")
            if background is None:
                raise ValueError("SHAP KernelExplainer fallback requires background_data.")
            explainer = shap.KernelExplainer(
                lambda x: adapter.predict_proba(model, x), np.asarray(background)
            )
            raw_values = explainer.shap_values(input_array)

        prediction_output = adapter.predict(model, input_data)
        prediction = int(
            prediction_output.numpy().argmax() if hasattr(prediction_output, "numpy")
            else np.asarray(prediction_output).argmax()
        )

        # Normalize SHAP's varying output shapes into one flat array for the predicted class
        if isinstance(raw_values, list):
            class_values = np.asarray(raw_values[prediction])[0]
        else:
            values_array = np.asarray(raw_values)
            class_values = values_array[0, :, prediction] if values_array.ndim == 3 else values_array[0]

        ranking = [
            {
                "feature": feature_names[i] if feature_names and i < len(feature_names) else f"feature_{i}",
                "importance": float(score),
            }
            for i, score in enumerate(class_values.flatten())
        ]
        ranking.sort(key=lambda item: abs(item["importance"]), reverse=True)

        return ExplanationResult(
            technique_used=self.name,
            prediction=prediction,
            feature_ranking=ranking,
            raw_output=class_values.tolist(),
            metadata={"framework": adapter.framework, "shap_variant": explainer.__class__.__name__},
        )