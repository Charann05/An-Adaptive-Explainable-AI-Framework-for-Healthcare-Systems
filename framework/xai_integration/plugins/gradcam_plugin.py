from ..base_plugin import XAIPlugin
from ..output_schema import ExplanationResult
from ..plugin_registry import register_plugin
from ..adapters import resolve_adapter


@register_plugin("GradCAM")
class GradCAMPlugin(XAIPlugin):
    name = "GradCAM"

    def is_compatible(self, model_info: dict) -> bool:
        return model_info.get("category") == "CNN" and model_info.get("input_type") == "Image"

    def explain(self, model, input_data, model_info: dict, adapter=None) -> ExplanationResult:
        if adapter is None:
            adapter = resolve_adapter(model, model_info)

        if not adapter.supports_gradients():
            raise ValueError(f"{adapter.framework} adapter does not support Grad-CAM.")

        target_layer, target_layer_name = adapter.get_last_conv_layer(model)
        output = adapter.predict(model, input_data)

        # Framework-agnostic: argmax works the same on torch tensors and numpy/tf arrays
        prediction = int(output.numpy().argmax()) if hasattr(output, "numpy") else int(output.argmax())
        probs = adapter.predict_proba(model, input_data)
        confidence = float(probs.numpy().max() * 100) if hasattr(probs, "numpy") else float(probs.max() * 100)

        heatmap = adapter.run_gradcam(model, input_data, target_layer, prediction)

        return ExplanationResult(
            technique_used=self.name,
            prediction=prediction,
            confidence_score=confidence,
            raw_output=heatmap,
            metadata={
                "framework": adapter.framework,
                "architecture": model_info.get("architecture"),
                "target_layer": target_layer_name,
            },
        )