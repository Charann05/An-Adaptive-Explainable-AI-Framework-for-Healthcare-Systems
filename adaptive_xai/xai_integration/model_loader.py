import importlib
import importlib.util
import inspect
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

from ..dependency_manager import prepare_model_dependencies
from .exceptions import ModelCompatibilityError, ModelLoadError


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
        if framework_key not in self.SUPPORTED_FRAMEWORKS:
            raise ModelLoadError(
                f"Unsupported model framework: {framework}"
            )

        dependency_result = prepare_model_dependencies(
            framework_key,
            path,
        )
        failed_dependencies = dependency_result["failed"]
        if failed_dependencies:
            packages = ", ".join(failed_dependencies)
            raise ModelLoadError(
                f"Could not install required dependencies for "
                f"'{framework}': {packages}."
            )

        try:
            if framework_key == "pytorch":
                return self._load_pytorch(path, model_factory)

            if framework_key in {
                "sklearn",
                "scikit-learn",
            }:
                try:
                    return self._load_joblib(
                        path,
                        sklearn_compatibility=True,
                    )
                except ModuleNotFoundError as exc:
                    if exc.name not in {"xgboost", "xgboost.sklearn"}:
                        raise

                    dependency_result = prepare_model_dependencies(
                        "xgboost",
                        path,
                    )
                    failed_dependencies = dependency_result["failed"]
                    if failed_dependencies:
                        packages = ", ".join(failed_dependencies)
                        raise ModelLoadError(
                            "This model requires XGBoost, but its dependency "
                            f"could not be prepared: {packages}."
                        ) from exc

                    return self._load_joblib(
                        path,
                        sklearn_compatibility=True,
                    )

            if framework_key == "xgboost":
                return self._load_xgboost(path)

            if framework_key in {
                "tensorflow",
                "keras",
            }:
                return self._load_tensorflow(path)

        except ModelCompatibilityError:
            raise
        except Exception as exc:
            if (
                framework_key in {"sklearn", "scikit-learn"}
                and isinstance(exc, ValueError)
                and "node array from the pickle has an incompatible dtype"
                in str(exc).lower()
            ):
                raise ModelCompatibilityError(
                    "This scikit-learn decision-tree model was saved with an "
                    "incompatible version. Its internal tree format differs "
                    "from the installed scikit-learn version."
                ) from exc
            raise ModelLoadError(
                f"Failed to load model '{path}' "
                f"using framework '{framework}': {exc}"
            ) from exc

        raise ModelLoadError(f"Unsupported model framework: {framework}")

    @staticmethod
    def _normalize_framework(framework: str) -> str:

        if not framework:
            raise ModelLoadError(
                "Model framework was not provided."
            )

        return adaptive_xai.strip().lower()

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
    def _load_joblib(
        path: Path,
        *,
        sklearn_compatibility: bool = False,
    ):

        import joblib

        if sklearn_compatibility:
            legacy_modules = {
                "sklearn.ensemble.forest": "sklearn.ensemble._forest",
                "sklearn.tree.tree": "sklearn.tree._classes",
            }
            for legacy_module, current_module in legacy_modules.items():
                try:
                    importlib.import_module(legacy_module)
                except ModuleNotFoundError as exc:
                    if exc.name != legacy_module:
                        raise
                    sys.modules[legacy_module] = importlib.import_module(
                        current_module
                    )
        def load():
            try:
                return joblib.load(path)
            except ModuleNotFoundError as exc:
                if (
                    not sklearn_compatibility
                    or exc.name != "xgboost.sklearn"
                ):
                    raise
                ModelLoader._ensure_legacy_xgboost_sklearn_module()
                return joblib.load(path)

        if not sklearn_compatibility:
            return load()

        from sklearn.exceptions import InconsistentVersionWarning

        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always", InconsistentVersionWarning)
            try:
                model = load()
            except Exception as exc:
                version_warnings = [
                    item
                    for item in captured_warnings
                    if issubclass(item.category, InconsistentVersionWarning)
                ]
                if not version_warnings:
                    raise

                saved_versions = tuple(sorted({
                    item.message.original_sklearn_version
                    for item in version_warnings
                }))
                installed_version = (
                    version_warnings[0].message.current_sklearn_version
                )
                versions = ", ".join(saved_versions)
                raise ModelCompatibilityError(
                    "The model records scikit-learn version "
                    f"{versions} when it was saved; the installed version is "
                    f"{installed_version}. Loading failed because the "
                    "serialized model uses an incompatible scikit-learn "
                    "format.",
                    package="scikit-learn",
                    saved_versions=saved_versions,
                    installed_version=installed_version,
                ) from exc

        for item in captured_warnings:
            warnings.showwarning(
                item.message,
                item.category,
                item.filename,
                item.lineno,
                file=item.file,
                line=item.line,
            )
        return model

    @staticmethod
    def _ensure_legacy_xgboost_sklearn_module() -> None:
        legacy_module = "xgboost.sklearn"
        try:
            importlib.import_module(legacy_module)
            return
        except ModuleNotFoundError as exc:
            if exc.name != legacy_module:
                raise

        xgboost = importlib.import_module("xgboost")
        estimator_names = (
            "XGBModel",
            "XGBClassifier",
            "XGBRegressor",
            "XGBRanker",
            "XGBRFClassifier",
            "XGBRFRegressor",
        )
        estimators = {
            name: getattr(xgboost, name)
            for name in estimator_names
            if hasattr(xgboost, name)
        }
        if not estimators:
            raise ModelLoadError(
                "The installed XGBoost package does not expose any supported "
                "scikit-learn estimator classes. Reinstall XGBoost with its "
                "scikit-learn interface in this Python environment, or use "
                "the version that created the model."
            )

        compatibility_module = ModuleType(legacy_module)
        compatibility_module.__dict__.update(estimators)
        sys.modules[legacy_module] = compatibility_module
        setattr(xgboost, "sklearn", compatibility_module)

    @staticmethod
    def _load_tensorflow(path: Path):

        import tensorflow as tf

        return tf.keras.models.load_model(
            path,
            custom_objects=_keras_legacy_custom_objects(tf),
            compile=False,
        )

    @staticmethod
    def _load_xgboost(path: Path):
        if path.suffix.lower() in {".pkl", ".pickle", ".joblib"}:
            return ModelLoader._load_joblib(path)

        import xgboost as xgb

        model = xgb.XGBModel()
        model.load_model(path)
        return model


model_loader = ModelLoader()


def _keras_legacy_custom_objects(tf: Any) -> dict[str, Any]:
    """Adapt legacy SeparableConv2D configs to current Keras arguments."""
    keras_separable_conv2d = tf.keras.layers.SeparableConv2D

    class CompatibleSeparableConv2D(keras_separable_conv2d):
        @classmethod
        def from_config(cls, config):
            config = dict(config)
            groups = config.pop("groups", 1)
            if groups != 1:
                raise ValueError(
                    "Cannot load SeparableConv2D with groups other than 1."
                )

            for legacy_name in (
                "kernel_initializer",
                "kernel_regularizer",
                "kernel_constraint",
            ):
                legacy_value = config.pop(legacy_name, None)
                if legacy_value is None:
                    continue
                suffix = legacy_name.removeprefix("kernel_")
                for operation in ("depthwise", "pointwise"):
                    name = f"{operation}_{suffix}"
                    if config.get(name) is None:
                        config[name] = legacy_value

            return super().from_config(config)

    return {"SeparableConv2D": CompatibleSeparableConv2D}


def load_model(
    model_path: str | Path,
    framework: str,
    model_factory: str | None = None,
):
    """
    Public helper used by the rest of the adaptive_xai.
    """

    return model_loader.load(
        model_path=model_path,
        framework=framework,
        model_factory=model_factory,
    )