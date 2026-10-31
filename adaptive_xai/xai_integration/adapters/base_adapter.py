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

    def run_gradcam(
        self,
        model: Any,
        input_data: Any,
        target_layer: Any,
        prediction: int,
    ) -> Any:
        """
        Run Grad-CAM for this adaptive_xai.

        Framework-specific gradient implementations belong in concrete
        adapters, not in this base class.
        """
        raise NotImplementedError(
            f"{self.framework} adapter does not implement Grad-CAM execution."
        )