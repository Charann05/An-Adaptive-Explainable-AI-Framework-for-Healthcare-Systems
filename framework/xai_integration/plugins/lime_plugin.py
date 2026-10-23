# framework/xai_integration/plugins/lime_plugin.py
from ..base_plugin import XAIPlugin
from ..output_schema import ExplanationResult
from ..plugin_registry import register_plugin
from ..adapters import resolve_adapter


@register_plugin("LIME")
class LIMEPlugin(XAIPlugin):

    name = "LIME"

    def is_compatible(self, model_info: dict) -> bool:
        return model_info.get("input_type") in ("Tabular", "Image", "Text")

    def explain(self, model, input_data, model_info: dict, adapter=None) -> ExplanationResult:
        import numpy as np

        if adapter is None:
            adapter = resolve_adapter(model, model_info)

        input_type = model_info.get("input_type")

        if input_type == "Tabular":
            return self._explain_tabular(model, input_data, model_info, adapter)
        elif input_type == "Image":
            return self._explain_image(model, input_data, model_info, adapter)
        elif input_type == "Text":
            return self._explain_text(model, input_data, model_info, adapter)

        raise ValueError(f"LIME does not support input_type={input_type}")

    def _explain_tabular(self, model, input_data, model_info, adapter):
        import numpy as np
        from lime.lime_tabular import LimeTabularExplainer

        training_data = model_info.get("background_data")
        if training_data is None:
            raise ValueError("LIME requires background_data for tabular input.")

        training_data = np.asarray(training_data)
        instance = np.asarray(input_data)[0]
        feature_names = model_info.get("feature_names")
        is_regression = model_info.get("task") == "Regression" or model_info.get(
            "category"
        ) in {"Linear", "Regression"}

        def predict_fn(values):
            predictions = adapter.predict(model, values)
            values = np.asarray(
                predictions.detach().cpu().numpy()
                if hasattr(predictions, "detach")
                else predictions
            )
            if is_regression:
                return values.reshape(-1)
            if values.ndim == 1:
                classes = np.unique(values)
                return np.column_stack([values == label for label in classes])
            return values

        explainer = LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            mode="regression" if is_regression else "classification",
            discretize_continuous=True,
        )
        explanation = explainer.explain_instance(
            instance,
            predict_fn,
            num_features=len(instance),
        )

        ranking = [{"feature": name, "importance": float(score)} for name, score in explanation.as_list()]
        prediction = adapter.predict(model, np.asarray(input_data))
        if hasattr(prediction, "detach"):
            prediction = prediction.detach().cpu().numpy()
        prediction = np.asarray(prediction).reshape(-1)[0]
        if not is_regression:
            prediction = int(prediction)

        return ExplanationResult(
            technique_used=self.name, prediction=prediction,
            feature_ranking=ranking, raw_output=explanation.as_list(),
        )

    def _explain_image(self, model, input_data, model_info, adapter):
        import numpy as np
        from lime.lime_image import LimeImageExplainer

        def predict_fn(images_batch):
            batch_input = np.asarray(images_batch)
            channel_order = model_info.get("channel_order")
            if channel_order is None:
                channel_order = (
                    "channels_first"
                    if adapter.framework == "pytorch"
                    else "channels_last"
                )
            if channel_order == "channels_first":
                batch_input = np.transpose(batch_input, (0, 3, 1, 2))
            output = adapter.predict_proba(model, batch_input)
            return (
                output.detach().cpu().numpy()
                if hasattr(output, "detach")
                else np.asarray(output)
            )

        image = np.asarray(input_data[0])
        if hasattr(image, "detach"):
            image = image.detach().numpy()
        if image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))

        explainer = LimeImageExplainer()
        explanation = explainer.explain_instance(image, predict_fn, top_labels=1, num_samples=100)

        prediction = explanation.top_labels[0]
        _, mask = explanation.get_image_and_mask(prediction, positive_only=True, num_features=5)

        return ExplanationResult(
            technique_used=self.name, prediction=int(prediction),
            raw_output=mask.tolist(),
            metadata={"note": "raw_output is a segmentation mask highlighting influential regions"},
        )

    def _explain_text(self, model, input_data, model_info, adapter):
        from lime.lime_text import LimeTextExplainer

        texts = input_data if isinstance(input_data, list) else [input_data]

        def predict_fn(values):
            output = adapter.predict_proba(model, values)
            return output.detach().cpu().numpy() if hasattr(output, "detach") else output

        explanation = LimeTextExplainer(
            class_names=model_info.get("class_names"),
        ).explain_instance(texts[0], predict_fn)
        prediction = adapter.predict(model, texts)
        if hasattr(prediction, "detach"):
            prediction = prediction.detach().cpu().numpy()
        return ExplanationResult(
            technique_used=self.name,
            prediction=int(prediction[0]),
            feature_ranking=[
                {"feature": word, "importance": float(score)}
                for word, score in explanation.as_list()
            ],
            raw_output=explanation.as_list(),
        )