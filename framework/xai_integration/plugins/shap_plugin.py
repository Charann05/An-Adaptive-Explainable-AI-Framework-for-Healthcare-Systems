from ..base_plugin import XAIPlugin
from ..output_schema import ExplanationResult
from ..plugin_registry import register_plugin
from ..adapters import resolve_adapter
from ..prediction_utils import predicted_value


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
        is_regression = model_info.get("task") == "Regression" or model_info.get(
            "category"
        ) in {"Linear", "Regression"}
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

            def predict_fn(values):
                try:
                    output = adapter.predict_proba(model, values)
                except NotImplementedError:
                    output = adapter.predict(model, values)
                if hasattr(output, "detach"):
                    output = output.detach().cpu().numpy()
                return np.asarray(output).reshape(-1) if is_regression else output

            explainer = shap.KernelExplainer(
                predict_fn, np.asarray(background)
            )
            raw_values = explainer.shap_values(input_array)

        prediction_output = adapter.predict(model, input_data)
        if hasattr(prediction_output, "detach"):
            prediction_output = prediction_output.detach().cpu().numpy()
        prediction = predicted_value(
            prediction_output,
            "Regression" if is_regression else "Classification",
            class_scores=adapter.framework in {"pytorch", "tensorflow"},
        )

        # Normalize SHAP's varying output shapes into one flat array for the predicted class
        if is_regression:
            class_values = np.asarray(raw_values).reshape(-1)
            if class_values.ndim > 1:
                class_values = class_values[0]
        elif isinstance(raw_values, list):
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