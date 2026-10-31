from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ENTRYPOINT_FILENAME = "adap_xai.py"


@dataclass(frozen=True)
class IntegrationManifest:
    model_path: str
    model_info: dict[str, Any]
    selector_result: dict[str, Any]
    technique: str
    model_factory: str | None = None


class PersistentIntegration:
    """Creates and loads the single user-owned integration entry point."""

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).expanduser().resolve()
        self.entrypoint = self.project_root / ENTRYPOINT_FILENAME
        self.artifact_root = self.project_root
        self.manifest_path = self.entrypoint

    def exists(self) -> bool:
        return self.entrypoint.is_file()

    def load_manifest(self) -> IntegrationManifest:
        if not self.exists():
            raise FileNotFoundError(
                f"No Adaptive XAI integration exists in {self.project_root}."
            )
        tree = ast.parse(self.entrypoint.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "INTEGRATION"
                for target in node.targets
            ):
                payload = ast.literal_eval(node.value)
                return IntegrationManifest(
                    model_path=payload["model_path"],
                    model_info=payload["model_info"],
                    selector_result=payload["selector_result"],
                    technique=payload["technique"],
                    model_factory=payload.get("model_factory"),
                )
        raise ValueError(f"Invalid Adaptive XAI entry point: {self.entrypoint}")

    def create(
        self,
        model_path: str | Path,
        model_info: dict[str, Any],
        selector_result: dict[str, Any],
    ) -> IntegrationManifest:
        model_info = self._resolve_feature_names(model_path, model_info)
        model_info, selector_result = self._resolve_runtime_capabilities(
            model_path,
            model_info,
            selector_result,
        )
        model_info = self._resolve_background_data(
            model_info,
            selector_result,
        )
        technique = selector_result.get("selected")
        if not technique:
            raise ValueError("Cannot integrate without a selected XAI technique.")

        manifest = IntegrationManifest(
            model_path=str(Path(model_path).expanduser().resolve()),
            model_info=dict(model_info),
            selector_result=dict(selector_result),
            technique=str(technique),
            model_factory=self._find_model_factory(model_path, model_info),
        )
        self._write_entrypoint(manifest)
        return manifest

    def _resolve_background_data(
        self,
        model_info: dict[str, Any],
        selector_result: dict[str, Any],
    ) -> dict[str, Any]:
        info = dict(model_info)
        if (
            info.get("input_type", "").strip().lower() != "tabular"
            or info.get("background_data") is not None
        ):
            return info

        alternatives = selector_result.get("alternatives") or []
        techniques = {selector_result.get("selected"), *alternatives}
        if not techniques.intersection({"SHAP", "LIME"}):
            return info

        import pandas as pd

        excluded_directories = {
            ".git",
            ".venv",
            "venv",
            "env",
            "__pycache__",
            "node_modules",
        }
        dataset_paths = [
            self.project_root / relative_path
            for relative_path in self._collect_dataset_paths(
                excluded_directories
            )
        ]
        dataset_paths.sort(
            key=lambda path: (
                not any(
                    token in path.stem.lower()
                    for token in ("train", "training", "dataset", "data")
                ),
                str(path).lower(),
            )
        )

        feature_names = info.get("feature_names")
        for dataset_path in dataset_paths:
            try:
                if dataset_path.suffix.lower() == ".csv":
                    dataframe = pd.read_csv(dataset_path)
                else:
                    dataframe = pd.read_excel(dataset_path)
            except (OSError, ValueError, ImportError, UnicodeDecodeError):
                continue

            if dataframe.empty:
                continue

            columns_by_name = {
                str(column).strip().casefold(): column
                for column in dataframe.columns
            }
            if feature_names:
                normalized_features = [
                    str(name).strip().casefold()
                    for name in feature_names
                ]
                if not all(name in columns_by_name for name in normalized_features):
                    continue
                selected_columns = [
                    columns_by_name[name] for name in normalized_features
                ]
            else:
                from ..model_analyzer.read_dataset import detect_target_column

                target_column = detect_target_column(list(dataframe.columns))
                selected_columns = [
                    column
                    for column in dataframe.columns
                    if column != target_column
                ]
                expected_features = info.get("num_inputs")
                if expected_features and len(selected_columns) != expected_features:
                    continue

            sample = dataframe.loc[:, selected_columns].sample(
                n=min(100, len(dataframe)),
                random_state=0,
            )
            sample = sample.astype(object).where(pd.notna(sample), None)
            info["background_data"] = sample.values.tolist()
            return info

        return info

    def _collect_dataset_paths(
        self,
        excluded_directories: set[str],
    ) -> list[str]:
        dataset_extensions = {".csv", ".xls", ".xlsx"}
        paths = []
        for path in self.project_root.rglob("*"):
            if (
                not path.is_file()
                or path.suffix.lower() not in dataset_extensions
                or any(part in excluded_directories for part in path.parts)
            ):
                continue
            paths.append(str(path.relative_to(self.project_root)))
        return paths

    def _resolve_feature_names(
        self,
        model_path: str | Path,
        model_info: dict[str, Any],
    ) -> dict[str, Any]:
        info = dict(model_info)
        existing = info.get("feature_names")
        if isinstance(existing, list) and existing:
            return info
        if info.get("input_type", "").strip().lower() != "tabular":
            return info

        framework = info.get("framework", "").strip().lower()
        if framework not in {"sklearn", "scikit-learn", "xgboost"}:
            return info

        from .model_loader import load_model

        model = load_model(model_path, framework)
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is None and hasattr(model, "get_booster"):
            feature_names = model.get_booster().feature_names
        if feature_names is not None:
            info["feature_names"] = [str(name) for name in feature_names]
        return info

    def _resolve_runtime_capabilities(
        self,
        model_path: str | Path,
        model_info: dict[str, Any],
        selector_result: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Use checkpoint/factory evidence when static analysis is unreliable."""
        info = dict(model_info)
        selection = dict(selector_result)
        factory = self._find_model_factory(model_path, info)
        if factory is None:
            return info, selection

        factory_source = Path(factory.rsplit(":", 1)[0]).read_text(
            encoding="utf-8"
        ).lower()
        if info.get("framework", "").strip().lower() == "pytorch" and (
            "conv" in factory_source or "efficientnet" in factory_source
        ):
            info.update({
                "architecture": "CNN",
                "category": "CNN",
                "input_type": "Image",
            })
            selection["selected"] = "GradCAM"
            selection["alternatives"] = ["SHAP", "LIME"]
        return info, selection

    def _write_entrypoint(self, manifest: IntegrationManifest) -> None:
        configuration = repr(asdict(manifest))
        self.entrypoint.write_text(
            f'''"""User-owned Adaptive XAI model and explanation entry point.

This is the only generated integration file. The framework implementation is
imported from the active Python environment.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from framework.xai_integration import load_model, run_explanation
from framework.xai_integration.exceptions import (
    ExplanationExecutionError,
    ModelLoadError,
)
from framework.xai_integration.visualization import display_gradcam
from framework.explanation_generator.generator import format_explanation
from framework.cli_display import print_explanation_error, print_model_load_error


INTEGRATION = {configuration}
VISUALIZATION_PATH = Path(__file__).with_name("gradcam_explanation.png")
if __name__ == "__main__":
    VISUALIZATION_PATH.unlink(missing_ok=True)
# Change this path to use a different model file.
MODEL_PATH = Path(INTEGRATION["model_path"])
# Replace this with representative tabular rows if automatic discovery missed
# your dataset. SHAP and LIME use this data as explanation background.
BACKGROUND_DATA = INTEGRATION["model_info"].get("background_data")
if not MODEL_PATH.is_absolute():
    MODEL_PATH = Path(__file__).parent / MODEL_PATH
try:
    _MODEL = load_model(
        MODEL_PATH,
        INTEGRATION["model_info"]["framework"],
        INTEGRATION["model_factory"],
    )
except ModelLoadError as error:
    print_model_load_error(error)
    raise SystemExit(1)
_CONFIG = None
_CONFIG_PATH = Path(__file__).with_name("adaptive_xai_config.py")
if _CONFIG_PATH.is_file():
    spec = importlib.util.spec_from_file_location("adaptive_xai_config", _CONFIG_PATH)
    if spec is not None and spec.loader is not None:
        _CONFIG = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_CONFIG)

# Add the same preprocessing used when training the model here, if needed.
# Example:
#     from loader import scaler
#     return scaler.transform(input_data)
def preprocess_input(input_data: Any):
    return input_data


def _apply_preprocessing(input_data: Any):
    input_data = preprocess_input(input_data)
    configured_preprocess = getattr(_CONFIG, "preprocess", None)
    if callable(configured_preprocess):
        input_data = configured_preprocess(input_data)
    return input_data


def _jsonable(value):
    if hasattr(value, "shape") and getattr(value, "shape", None) != ():
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {{key: _jsonable(item) for key, item in value.items()}}
    return value


def _load_image(path: str):
    framework = INTEGRATION["model_info"].get("framework", "").strip().lower()
    if framework == "pytorch":
        try:
            from src.data import basic_preprocess, get_transforms
            import cv2

            image = cv2.imread(path)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {{path}}")
            image = image[:, :, ::-1]
            image = basic_preprocess(image)
            return get_transforms(train=False)(image=image)["image"].unsqueeze(0)
        except ImportError:
            pass

    from PIL import Image
    import numpy as np

    image = Image.open(path).convert("RGB").resize((224, 224))
    array = np.asarray(image, dtype=np.float32) / 255.0
    if framework in {{"tensorflow", "keras"}}:
        return array[np.newaxis, ...]
    import torch

    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)


def _prompt_input():
    input_type = INTEGRATION["model_info"].get("input_type", "").lower()
    if input_type == "image":
        return _load_image(input("Enter the image path: ").strip())
    if input_type == "text":
        return input("Enter the text input: ")
    if input_type == "audio":
        path = input("Enter the audio file path: ").strip()
        if _CONFIG is None or not callable(getattr(_CONFIG, "preprocess", None)):
            raise ValueError(
                "Audio input requires adaptive_xai_config.py with "
                "a preprocess(input_data) function."
            )
        return path

    feature_names = INTEGRATION["model_info"].get("feature_names")
    if feature_names:
        values = []
        for name in feature_names:
            value = input(f"Enter {{name}}: ").strip()
            if not value:
                raise ValueError(f"Value for {{name}} cannot be empty.")
            try:
                values.append(json.loads(value))
            except json.JSONDecodeError:
                values.append(value)
        return [values]

    value = input(
        "Enter the model input values separated by commas "
        "(or enter a JSON value): "
    ).strip()
    if not value:
        raise ValueError("Model input cannot be empty.")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = json.loads("[" + value + "]")
    if input_type == "tabular" and parsed and not isinstance(parsed[0], list):
        return [parsed]
    return parsed


def run(input_data: Any):
    input_data = _apply_preprocessing(input_data)
    model_info = dict(INTEGRATION["model_info"])
    if BACKGROUND_DATA is not None:
        model_info["background_data"] = _apply_preprocessing(BACKGROUND_DATA)
    result = run_explanation(
        model=_MODEL,
        input_data=input_data,
        model_info=model_info,
        selector_result=INTEGRATION["selector_result"],
    )
    return _jsonable({{
        "prediction": result.prediction,
        "technique_used": result.technique_used,
        "confidence_score": result.confidence_score,
        "feature_ranking": result.feature_ranking,
        "raw_output": result.raw_output,
        "metadata": result.metadata,
        "success": result.success,
        "error": result.error,
    }})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the integrated model and print its XAI result."
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input-json")
    inputs.add_argument("--input-image")
    args = parser.parse_args()
    if args.input_json is not None:
        input_data = json.loads(args.input_json)
    elif args.input_image is not None:
        input_data = _load_image(args.input_image)
    else:
        input_data = _prompt_input()
    try:
        output = run(input_data)
    except ExplanationExecutionError as error:
        print_explanation_error(error)
        raise SystemExit(1) from error
    if output["technique_used"] == "GradCAM":
        visualization_path = display_gradcam(
            input_data,
            output["raw_output"],
            VISUALIZATION_PATH,
            f"Grad-CAM explanation for prediction {{output['prediction']}}",
        )
        output["visualization"] = {{
            "type": "Grad-CAM overlay",
            "path": str(visualization_path),
        }}
    print(format_explanation(
        output,
        INTEGRATION["model_info"].get("input_type", ""),
        output.get("visualization", {{}}).get("path"),
    ))
''',
            encoding="utf-8",
        )

    def _find_model_factory(
        self,
        model_path: str | Path,
        model_info: dict[str, Any],
    ) -> str | None:
        if model_info.get("framework", "").strip().lower() != "pytorch":
            return None
        path = Path(model_path).expanduser().resolve()
        if path.suffix.lower() not in {".pt", ".pth", ".bin"}:
            return None

        candidates = [
            self.project_root / "src" / "model.py",
            self.project_root / "model.py",
        ]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            source = candidate.read_text(encoding="utf-8")
            if "def create_model" in source:
                return f"{candidate}:create_model"
        return None


def find_model_files(project_root: str | Path, framework: str) -> list[Path]:
    root = Path(project_root).expanduser().resolve()
    normalized = framework.strip().lower()
    extensions = {
        "pytorch": {".pt", ".pth", ".bin"},
        "tensorflow": {".keras", ".h5", ".hdf5"},
        "keras": {".keras", ".h5", ".hdf5"},
        "sklearn": {".pkl", ".pickle", ".joblib"},
        "scikit-learn": {".pkl", ".pickle", ".joblib"},
        "xgboost": {".pkl", ".pickle", ".json", ".ubj"},
    }.get(normalized, {".pkl", ".pickle", ".joblib", ".pt", ".pth", ".keras", ".h5"})
    excluded = {".git", ".venv", "venv", "env", "node_modules", "__pycache__"}
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name != ENTRYPOINT_FILENAME
        and path.suffix.lower() in extensions
        and not any(part in excluded for part in path.parts)
    ]
    candidates.sort(
        key=lambda path: (
            "model" not in path.stem.lower(),
            str(path).lower(),
        )
    )
    return candidates


def find_model_file(project_root: str | Path, framework: str) -> Path:
    candidates = find_model_files(project_root, framework)
    if not candidates:
        root = Path(project_root).expanduser().resolve()
        raise FileNotFoundError(
            f"Could not find a {framework} model file in {root}. "
            "Pass the model path to integrate a specific model."
        )
    return candidates[0]
