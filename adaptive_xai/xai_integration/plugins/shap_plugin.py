from ..base_plugin import XAIPlugin
from ..output_schema import ExplanationResult
from ..plugin_registry import register_plugin
from ..adapters import resolve_adapter
from ..prediction_utils import predicted_value


def _class_output_index(model, prediction, output_count: int) -> int:
    import numpy as np

    if output_count == 1:
        return 0

    classes = getattr(model, "classes_", None)
    if isinstance(classes, (list, tuple)) and classes and isinstance(
        classes[0], (list, tuple, np.ndarray)
    ):
        classes = classes[0]
    if classes is not None:
        matches = np.flatnonzero(np.asarray(classes) == prediction)
        if matches.size:
            return int(matches[0])

    if isinstance(prediction, (int, np.integer)):
        class_index = int(prediction)
        if 0 <= class_index < output_count:
            return class_index
    raise ValueError(
        "Could not map the model's predicted class to SHAP's output index."
    )


def _select_shap_values(raw_values, prediction, model, feature_count, is_regression):
    import numpy as np

    if hasattr(raw_values, "values"):
        raw_values = raw_values.values

    if isinstance(raw_values, list):
        output_count = len(raw_values)
        if is_regression or output_count == 1:
            selected = np.asarray(raw_values[0]).reshape(-1, feature_count)[0]
        else:
            output_index = _class_output_index(model, prediction, output_count)
            selected = np.asarray(raw_values[output_index]).reshape(
                -1, feature_count
            )[0]
    else:
        values = np.asarray(raw_values)
        if values.ndim == 1 and values.size == feature_count:
            selected = values
        elif values.ndim == 2 and values.shape[-1] == feature_count:
            selected = values[0]
        elif values.ndim == 3:
            if values.shape[1] == feature_count:
                output_count = values.shape[2]
                output_index = (
                    0
                    if is_regression
                    else _class_output_index(model, prediction, output_count)
                )
                selected = values[0, :, output_index]
            elif values.shape[2] == feature_count:
                output_count = values.shape[1]
                output_index = (
                    0
                    if is_regression
                    else _class_output_index(model, prediction, output_count)
                )
                selected = values[0, output_index, :]
            else:
                raise ValueError(
                    "SHAP output dimensions do not match the model's feature count."
                )
        else:
            raise ValueError(
                "Unsupported SHAP output shape for a single model input: "
                f"{values.shape}."
            )

    selected = np.asarray(selected).reshape(-1)
    if selected.size != feature_count:
        raise ValueError(
            "SHAP returned a different number of feature scores than the "
            "model input."
        )
    return selected


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
            background_data = model_info.get("background_data")
            if background_data is None:
                raise ValueError(
                    "SHAP DeepExplainer requires representative "
                    "background_data distinct from the explained input."
                )
            background = torch.as_tensor(background_data).float()
            explanation_input = torch.as_tensor(input_array).float()
            explainer = shap.DeepExplainer(model, background)
            raw_values = explainer.shap_values(explanation_input)

        elif adapter.framework == "tensorflow":
            background = model_info.get("background_data")
            if background is None:
                raise ValueError(
                    "SHAP GradientExplainer requires representative "
                    "background_data distinct from the explained input."
                )
            explainer = shap.GradientExplainer(model, np.asarray(background))
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

        class_values = _select_shap_values(
            raw_values,
            prediction,
            model,
            input_array.shape[-1],
            is_regression,
        )

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
            metadata={
                "framework": adapter.framework,
                "shap_variant": explainer.__class__.__name__,
            },
        )