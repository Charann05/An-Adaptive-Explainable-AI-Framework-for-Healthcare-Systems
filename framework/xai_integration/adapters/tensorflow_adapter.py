from typing import Any

from .base_adapter import ModelAdapter
from .adapter_registry import register_adapter


@register_adapter
class TensorFlowAdapter(ModelAdapter):

    framework = "tensorflow"

    def can_handle(
        self,
        model: Any,
        model_info: dict,
    ) -> bool:

        try:
            import tensorflow as tf

            return isinstance(
                model,
                tf.keras.Model,
            )

        except ImportError:
            return False

    def predict(
        self,
        model: Any,
        input_data: Any,
    ) -> Any:

        return model(
            input_data,
            training=False,
        )

    def predict_proba(
        self,
        model: Any,
        input_data: Any,
    ) -> Any:

        return self.predict(
            model,
            input_data,
        )

    def get_layers(
        self,
        model: Any,
    ) -> list[tuple[str, Any]]:

        return [
            (layer.name, layer)
            for layer in model.layers
        ]

    def get_last_conv_layer(
        self,
        model: Any,
    ) -> tuple[Any, str]:

        for layer in reversed(
            model.layers
        ):

            class_name = (
                layer.__class__
                .__name__
                .lower()
            )

            if "conv" in class_name:
                return (
                    layer,
                    layer.name,
                )

        raise ValueError(
            "No convolutional layer found "
            "in TensorFlow/Keras model."
        )

    def supports_gradients(self) -> bool:
        return True

    def run_gradcam(self, model, input_data, target_layer, prediction: int):
        import tensorflow as tf

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(input_data, training=False)
            loss = predictions[:, prediction]

        grads = tape.gradient(loss, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.math.reduce_max(heatmap) + 1e-8
        return heatmap.numpy()

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