from ..base_plugin import XAIPlugin
from ..output_schema import ExplanationResult
from ..plugin_registry import register_plugin
from ..adapters import resolve_adapter
from ..prediction_utils import predicted_class_and_confidence
import numpy as np


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

        prediction_output = adapter.predict_proba(model, input_data)
        prediction, confidence = predicted_class_and_confidence(prediction_output)
        confidence *= 100

        heatmap = adapter.run_gradcam(model, input_data, target_layer, prediction)
        heatmap_array = np.asarray(heatmap)
        if heatmap_array.ndim == 3:
            heatmap_array = heatmap_array[0]
        if heatmap_array.ndim != 2:
            raise ValueError(
                "Grad-CAM must return a two-dimensional spatial heatmap."
            )

        maximum = float(heatmap_array.max())
        minimum = float(heatmap_array.min())
        threshold = minimum + 0.5 * (maximum - minimum)
        active = heatmap_array >= threshold
        rows, columns = np.where(active)
        if rows.size:
            bbox = {
                "x_min": int(columns.min()),
                "y_min": int(rows.min()),
                "x_max": int(columns.max()),
                "y_max": int(rows.max()),
            }
        else:
            bbox = None
        peak_y, peak_x = np.unravel_index(
            int(np.argmax(heatmap_array)),
            heatmap_array.shape,
        )
        height, width = heatmap_array.shape

        return ExplanationResult(
            technique_used=self.name,
            prediction=prediction,
            confidence_score=confidence,
            raw_output={
                "shape": list(np.asarray(heatmap).shape),
                "heatmap": heatmap_array,
                "min": minimum,
                "max": maximum,
                "mean": float(heatmap_array.mean()),
                "threshold": threshold,
                "peak": {
                    "x": int(peak_x),
                    "y": int(peak_y),
                    "x_normalized": float(peak_x / max(width - 1, 1)),
                    "y_normalized": float(peak_y / max(height - 1, 1)),
                },
                "active_region_bbox": bbox,
            },
            metadata={
                "framework": adapter.framework,
                "architecture": model_info.get("architecture"),
                "target_layer": target_layer_name,
                "heatmap_coordinate_system": "heatmap_pixels",
            },
        )