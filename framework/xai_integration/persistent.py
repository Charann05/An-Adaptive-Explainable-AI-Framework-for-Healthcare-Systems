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
from framework.xai_integration.visualization import display_gradcam


INTEGRATION = {configuration}
_MODEL = load_model(
    INTEGRATION["model_path"],
    INTEGRATION["model_info"]["framework"],
    INTEGRATION["model_factory"],
)
_CONFIG = None
_CONFIG_PATH = Path(__file__).with_name("adaptive_xai_config.py")
if _CONFIG_PATH.is_file():
    spec = importlib.util.spec_from_file_location("adaptive_xai_config", _CONFIG_PATH)
    if spec is not None and spec.loader is not None:
        _CONFIG = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_CONFIG)


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
        from PIL import Image
        import numpy as np
        import torch

        image = Image.open(path).convert("RGB").resize((224, 224))
        array = np.asarray(image, dtype=np.float32) / 255.0
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
    preprocess = getattr(_CONFIG, "preprocess", None)
    if callable(preprocess):
        input_data = preprocess(input_data)
    result = run_explanation(
        model=_MODEL,
        input_data=input_data,
        model_info=INTEGRATION["model_info"],
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
    output = run(input_data)
    if output["technique_used"] == "GradCAM":
        visualization_path = display_gradcam(
            input_data,
            output["raw_output"],
            Path(__file__).with_name("gradcam_explanation.png"),
            f"Grad-CAM explanation for prediction {{output['prediction']}}",
        )
        output["visualization"] = {{
            "type": "Grad-CAM overlay",
            "path": str(visualization_path),
        }}
    print(json.dumps(output, indent=2, default=str))
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


def find_model_file(project_root: str | Path, framework: str) -> Path:
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
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a {framework} model file in {root}. "
            "Pass the model path to integrate a specific model."
        )
    candidates.sort(key=lambda path: ("model" not in path.stem.lower(), str(path).lower()))
    return candidates[0]
