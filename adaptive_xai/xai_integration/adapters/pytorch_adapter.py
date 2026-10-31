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

        parameter = next(model.parameters(), None)
        dtype = (
            parameter.dtype
            if parameter is not None and parameter.is_floating_point()
            else torch.float32
        )
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.as_tensor(input_data, dtype=dtype)
        if parameter is not None:
            input_data = input_data.to(
                device=parameter.device,
                dtype=dtype,
            )
        elif not input_data.is_floating_point():
            input_data = input_data.to(dtype=dtype)

        model.eval()

        with torch.no_grad():
            return model(input_data)

    def predict_proba(
        self,
        model: Any,
        input_data: Any,
    ) -> Any:

        output = self.predict(
            model,
            input_data,
        )
        from ..prediction_utils import probability_matrix

        return probability_matrix(output)

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

        cam = GradCAM(model=model, target_layers=[target_layer])

        def target(model_output):
            scores = model_output.reshape(model_output.shape[0], -1)
            if scores.shape[1] == 1:
                score = scores[:, 0]
                return score if prediction == 1 else -score
            return scores[:, prediction]

        return cam(
            input_tensor=input_data,
            targets=[target],
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