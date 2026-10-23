from pathlib import Path
import importlib.util
import inspect
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
        model_factory: str | None = None,
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
                return self._load_pytorch(path, model_factory)

            if framework_key in {
                "sklearn",
                "scikit-learn",
            }:
                return self._load_joblib(path)

            if framework_key == "xgboost":
                return self._load_xgboost(path)

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
    def _load_pytorch(path: Path, model_factory: str | None = None):

        import torch

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        if model_factory is None:
            return checkpoint

        factory = ModelLoader._load_factory(model_factory)
        parameters = inspect.signature(factory).parameters
        factory_options = {}
        if "pretrained" in parameters:
            factory_options["pretrained"] = False
        model = factory(**factory_options)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(
            checkpoint, dict
        ) else checkpoint
        if not isinstance(state_dict, dict):
            raise ModelLoadError(
                "The PyTorch checkpoint is not a state_dict and cannot be "
                f"loaded with factory '{model_factory}'."
            )
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @staticmethod
    def _load_factory(factory_path: str):
        module_name, separator, function_name = factory_path.rpartition(":")
        if not separator or not module_name or not function_name:
            raise ModelLoadError(
                "Model factory must use the 'module.py:function' format."
            )
        module_path = Path(module_name).expanduser().resolve()
        spec = importlib.util.spec_from_file_location(
            "adaptive_xai_user_model_factory",
            module_path,
        )
        if spec is None or spec.loader is None:
            raise ModelLoadError(f"Could not load model factory: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        factory = getattr(module, function_name, None)
        if not callable(factory):
            raise ModelLoadError(
                f"Model factory function was not found: {factory_path}"
            )
        return factory


    @staticmethod
    def _load_joblib(path: Path):

        import joblib

        return joblib.load(path)

    @staticmethod
    def _load_tensorflow(path: Path):

        import tensorflow as tf

        return tf.keras.models.load_model(path)

    @staticmethod
    def _load_xgboost(path: Path):
        if path.suffix.lower() in {".pkl", ".pickle", ".joblib"}:
            return ModelLoader._load_joblib(path)

        import xgboost as xgb

        model = xgb.XGBModel()
        model.load_model(path)
        return model


model_loader = ModelLoader()


def load_model(
    model_path: str | Path,
    framework: str,
    model_factory: str | None = None,
):
    """
    Public helper used by the rest of the framework.
    """

    return model_loader.load(
        model_path=model_path,
        framework=framework,
        model_factory=model_factory,
    )