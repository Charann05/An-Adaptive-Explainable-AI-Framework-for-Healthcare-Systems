from pathlib import Path
from typing import Any

from .exceptions import ModelLoadError


class ModelLoader:

    SUPPORTED_FRAMEWORKS = {
        "pytorch",
        "sklearn",
        "scikit-learn",
        "xgboost",
        "tensorflow",
        "keras",
    }

    def load(
        self,
        model_path: str | Path,
        framework: str,
    ) -> Any:
        """
        Load a pretrained model using the framework detected
        by the Model Analyzer.
        """

        if not model_path:
            raise ModelLoadError(
                "Model path was not provided."
            )

        path = Path(model_path)

        if not path.exists():
            raise ModelLoadError(
                f"Model file does not exist: {path}"
            )

        if not path.is_file():
            raise ModelLoadError(
                f"Model path is not a file: {path}"
            )

        framework_key = self._normalize_framework(framework)

        try:
            if framework_key == "pytorch":
                return self._load_pytorch(path)

            if framework_key in {
                "sklearn",
                "scikit-learn",
                "xgboost",
            }:
                return self._load_joblib(path)

            if framework_key in {
                "tensorflow",
                "keras",
            }:
                return self._load_tensorflow(path)

        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load model '{path}' "
                f"using framework '{framework}': {exc}"
            ) from exc

        raise ModelLoadError(
            f"Unsupported model framework: {framework}"
        )

    @staticmethod
    def _normalize_framework(framework: str) -> str:

        if not framework:
            raise ModelLoadError(
                "Model framework was not provided."
            )

        return framework.strip().lower()

    @staticmethod
    def _load_pytorch(path: Path):

        import torch

        return torch.load(
            path,
            weights_only=False,
        )

    @staticmethod
    def _load_joblib(path: Path):

        import joblib

        return joblib.load(path)

    @staticmethod
    def _load_tensorflow(path: Path):

        import tensorflow as tf

        return tf.keras.models.load_model(path)


model_loader = ModelLoader()


def load_model(
    model_path: str | Path,
    framework: str,
):
    """
    Public helper used by the rest of the framework.
    """

    return model_loader.load(
        model_path=model_path,
        framework=framework,
    )