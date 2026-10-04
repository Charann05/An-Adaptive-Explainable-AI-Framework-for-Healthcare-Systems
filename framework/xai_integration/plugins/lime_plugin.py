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

        if not hasattr(model, "predict_proba"):
            raise ValueError("LIME tabular requires model.predict_proba().")

        training_data = model_info.get("background_data")
        if training_data is None:
            raise ValueError("LIME requires background_data for tabular input.")

        training_data = np.asarray(training_data)
        instance = np.asarray(input_data)[0]
        feature_names = model_info.get("feature_names")

        explainer = LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            mode="classification",
            discretize_continuous=True,
        )
        explanation = explainer.explain_instance(instance, model.predict_proba, num_features=len(instance))

        ranking = [{"feature": name, "importance": float(score)} for name, score in explanation.as_list()]
        prediction = int(model.predict(np.asarray(input_data))[0])

        return ExplanationResult(
            technique_used=self.name, prediction=prediction,
            feature_ranking=ranking, raw_output=explanation.as_list(),
        )

    def _explain_image(self, model, input_data, model_info, adapter):
        import numpy as np
        import torch
        from lime.lime_image import LimeImageExplainer

        def predict_fn(images_batch):
            batch_input = np.transpose(images_batch, (0, 3, 1, 2))
            batch_input = torch.as_tensor(batch_input).float().contiguous()
            output = adapter.predict_proba(model, batch_input)
            return output.detach().numpy() if hasattr(output, "detach") else np.asarray(output)

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