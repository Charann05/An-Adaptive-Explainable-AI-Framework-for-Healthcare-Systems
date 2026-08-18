from typing import Any

from .base_adapter import ModelAdapter
from .adapter_registry import register_adapter


@register_adapter
class SklearnAdapter(ModelAdapter):

    framework = "sklearn"

    def can_handle(
        self,
        model: Any,
        model_info: dict,
    ) -> bool:

        try:
            from sklearn.base import BaseEstimator

            return isinstance(
                model,
                BaseEstimator,
            )

        except ImportError:
            return False

    def predict(
        self,
        model: Any,
        input_data: Any,
    ) -> Any:

        return model.predict(
            input_data
        )

    def predict_proba(
        self,
        model: Any,
        input_data: Any,
    ) -> Any:

        if not hasattr(
            model,
            "predict_proba",
        ):
            raise NotImplementedError(
                "Model does not expose predict_proba()."
            )

        return model.predict_proba(
            input_data
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

        if hasattr(
            model,
            "coef_",
        ):
            return model.coef_

        raise NotImplementedError(
            "Model does not expose native "
            "feature importance."
        )

    def supports_tree_structure(
        self,
        model: Any,
    ) -> bool:

        return (
            hasattr(model, "tree_")
            or hasattr(model, "estimators_")
        )