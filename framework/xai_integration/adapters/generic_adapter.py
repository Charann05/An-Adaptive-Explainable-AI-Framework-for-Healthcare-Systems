from typing import Any

from .base_adapter import ModelAdapter
from .adapter_registry import register_adapter


@register_adapter
class GenericAdapter(ModelAdapter):
    """Fallback adapter for estimators exposing the common predict contract."""

    framework = "generic"

    def can_handle(self, model: Any, model_info: dict) -> bool:
        return callable(getattr(model, "predict", None))

    def predict(self, model: Any, input_data: Any) -> Any:
        return model.predict(input_data)

    def predict_proba(self, model: Any, input_data: Any) -> Any:
        if not callable(getattr(model, "predict_proba", None)):
            raise NotImplementedError(
                "The model does not expose predict_proba()."
            )
        return model.predict_proba(input_data)

    def get_feature_importances(self, model: Any) -> Any:
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        if hasattr(model, "coef_"):
            return model.coef_
        raise NotImplementedError(
            "The model does not expose native feature importance."
        )
