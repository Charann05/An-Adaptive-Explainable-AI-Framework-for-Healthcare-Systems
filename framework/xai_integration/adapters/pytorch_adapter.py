from typing import Any

from .base_adapter import ModelAdapter
from .adapter_registry import register_adapter


@register_adapter
class PyTorchAdapter(ModelAdapter):

    framework = "pytorch"

    def can_handle(
        self,
        model: Any,
        model_info: dict,
    ) -> bool:

        try:
            import torch.nn as nn

            return isinstance(model, nn.Module)

        except ImportError:
            return False

    def predict(
        self,
        model: Any,
        input_data: Any,
    ) -> Any:

        import torch

        model.eval()

        with torch.no_grad():
            return model(input_data)

    def predict_proba(
        self,
        model: Any,
        input_data: Any,
    ) -> Any:

        import torch

        output = self.predict(
            model,
            input_data,
        )

        return torch.softmax(
            output,
            dim=-1,
        )

    def get_layers(
        self,
        model: Any,
    ) -> list[tuple[str, Any]]:

        return list(
            model.named_modules()
        )

    def get_last_conv_layer(
        self,
        model: Any,
    ) -> tuple[Any, str]:

        import torch.nn as nn

        candidate = None

        for name, module in model.named_modules():

            if isinstance(
                module,
                (nn.Conv1d, nn.Conv2d, nn.Conv3d),
            ):
                candidate = (
                    module,
                    name,
                )

        if candidate is None:
            raise ValueError(
                "No convolutional layer found "
                "in PyTorch model."
            )

        return candidate

    def supports_gradients(self) -> bool:
        return True

    def run_gradcam(self, model, input_data, target_layer, prediction: int):
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        cam = GradCAM(model=model, target_layers=[target_layer])
        return cam(
            input_tensor=input_data,
            targets=[ClassifierOutputTarget(prediction)],
        )

    def get_feature_importances(
        self,
        model: Any,
    ) -> Any:

        if hasattr(
            model,
            "feature_importances_",
        ):
            return model.feature_importances_

        raise NotImplementedError(
            "Model does not expose native "
            "feature importance."
        )