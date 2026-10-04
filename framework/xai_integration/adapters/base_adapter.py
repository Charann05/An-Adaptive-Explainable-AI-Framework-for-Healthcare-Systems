from abc import ABC, abstractmethod
from typing import Any


class ModelAdapter(ABC):
    """
    Normalized interface between ML frameworks and XAI plugins.

    XAI plugins should use this interface instead of directly
    depending on TensorFlow, PyTorch, sklearn, etc. wherever possible.
    """

    framework: str = "unknown"

    @abstractmethod
    def can_handle(self, model: Any, model_info: dict) -> bool:
        """Return True if this adapter can handle the model."""
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        model: Any,
        input_data: Any,
    ) -> Any:
        """Return model predictions."""
        raise NotImplementedError

    def predict_proba(
        self,
        model: Any,
        input_data: Any,
    ) -> Any:
        """
        Return class probabilities when supported.
        """
        raise NotImplementedError(
            f"{self.framework} adapter does not provide "
            "probability prediction."
        )

    def get_layers(
        self,
        model: Any,
    ) -> list[tuple[str, Any]]:
        """
        Return model layers/modules when applicable.
        """
        return []

    def get_last_conv_layer(
        self,
        model: Any,
    ) -> tuple[Any, str]:
        """
        Return a suitable convolutional layer for Grad-CAM.
        """
        raise NotImplementedError(
            f"{self.framework} adapter does not provide "
            "convolutional-layer discovery."
        )

    def get_feature_importances(
        self,
        model: Any,
    ) -> Any:
        """
        Return native feature importance when available.
        """
        raise NotImplementedError(
            f"{self.framework} adapter does not expose "
            "native feature importance."
        )

    def supports_gradients(self) -> bool:
        """
        Whether gradient-based XAI techniques can be supported.
        """
        return False

    def supports_tree_structure(
        self,
        model: Any,
    ) -> bool:
        """
        Whether the underlying model exposes a compatible tree structure.
        """
        return False

def run_gradcam(self, model, input_data, target_layer, prediction: int):
    """
    Run Grad-CAM for this framework and return a grayscale heatmap array.
    Each framework computes gradients differently, so this lives per-adapter.
    """
    raise NotImplementedError(
        f"{self.framework} adapter does not implement Grad-CAM execution."
    )
def run_gradcam(self, model, input_data, target_layer, prediction: int):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    cam = GradCAM(model=model, target_layers=[target_layer])
    return cam(input_tensor=input_data, targets=[ClassifierOutputTarget(prediction)])
def run_gradcam(self, model, input_data, target_layer, prediction: int):
    import tensorflow as tf
    import numpy as np

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(input_data, training=False)
        loss = predictions[:, prediction]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()