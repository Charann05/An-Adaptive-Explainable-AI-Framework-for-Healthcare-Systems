from ..base_plugin import XAIPlugin
from ..output_schema import ExplanationResult
from ..plugin_registry import register_plugin
from ..adapters import resolve_adapter


@register_plugin("GradCAM")
class GradCAMPlugin(XAIPlugin):

    name = "GradCAM"

    def is_compatible(self, model_info: dict) -> bool:
        return (
            model_info.get("category") == "CNN"
            and model_info.get("input_type") == "Image"
        )

    def explain(self, model, input_data, model_info: dict, adapter=None) -> ExplanationResult:
        import torch
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        if adapter is None:
            adapter = resolve_adapter(model, model_info)

        if not adapter.supports_gradients():
            raise ValueError(
                f"{adapter.framework} adapter does not support gradient-based techniques like Grad-CAM."
            )

        target_layer, target_layer_name = adapter.get_last_conv_layer(model)
        output = adapter.predict(model, input_data)

        if output.ndim != 2:
            raise ValueError(
                "Grad-CAM currently expects a classification model producing output of shape [batch, classes]."
            )

        prediction = int(output.argmax(dim=1).item())
        confidence = float(torch.softmax(output, dim=1).max().item() * 100)

        cam = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=input_data, targets=[ClassifierOutputTarget(prediction)])

        return ExplanationResult(
            technique_used=self.name,
            prediction=prediction,
            confidence_score=confidence,
            raw_output=grayscale_cam,
            metadata={
                "framework": adapter.framework,
                "architecture": model_info.get("architecture"),
                "target_layer": target_layer_name,
            },
        )