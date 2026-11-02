from pathlib import Path
from typing import Any

import numpy as np

from .prediction_utils import to_numpy


def display_gradcam(
    input_data: Any,
    heatmap: Any,
    output_path: str | Path,
    title: str,
) -> Path:
    import matplotlib
    import matplotlib.pyplot as plt

    image = to_numpy(input_data)
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 2:
        image = np.repeat(image[..., np.newaxis], 3, axis=-1)
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(
            f"Expected a single image input, received shape {image.shape}."
        )

    image = np.nan_to_num(image.astype(np.float32, copy=False))
    minimum = float(image.min())
    maximum = float(image.max())
    if minimum < 0 or maximum > 1:
        image_range = maximum - minimum
        image = (
            (image - minimum) / image_range
            if image_range > 0
            else np.zeros_like(image)
        )

    attribution = to_numpy(heatmap).squeeze()
    if attribution.ndim != 2:
        raise ValueError(
            f"Expected a 2D Grad-CAM heatmap, received shape {attribution.shape}."
        )
    attribution = np.nan_to_num(attribution.astype(np.float32, copy=False))

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots()
    axis.imshow(image)
    overlay = axis.imshow(attribution, cmap="jet", alpha=0.4)
    axis.set_title(title)
    axis.axis("off")
    figure.colorbar(overlay, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    if matplotlib.get_backend().lower() not in {
        "agg",
        "cairo",
        "pdf",
        "ps",
        "svg",
        "template",
    }:
        plt.show()
    plt.close(figure)
    return destination


def display_lime_image(
    input_data: Any,
    mask: Any,
    output_path: str | Path,
    title: str,
) -> Path:
    import matplotlib
    import matplotlib.pyplot as plt

    image = to_numpy(input_data)
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 2:
        image = np.repeat(image[..., np.newaxis], 3, axis=-1)
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(
            f"Expected a single image input, received shape {image.shape}."
        )

    image = np.nan_to_num(image.astype(np.float32, copy=False))
    minimum = float(image.min())
    maximum = float(image.max())
    if minimum < 0 or maximum > 1:
        image_range = maximum - minimum
        image = (
            (image - minimum) / image_range
            if image_range > 0
            else np.zeros_like(image)
        )

    attribution = to_numpy(mask).squeeze()
    if attribution.ndim != 2 or attribution.shape != image.shape[:2]:
        raise ValueError(
            "Expected a 2D LIME mask matching the image dimensions; "
            f"received mask shape {attribution.shape} for image shape "
            f"{image.shape[:2]}."
        )
    attribution = attribution > 0
    if not attribution.any():
        raise ValueError("The LIME mask does not contain any highlighted regions.")

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots()
    axis.imshow(image)
    overlay = np.ma.masked_where(~attribution, attribution)
    axis.imshow(overlay, cmap="spring", alpha=0.45, vmin=0, vmax=1)
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    if matplotlib.get_backend().lower() not in {
        "agg",
        "cairo",
        "pdf",
        "ps",
        "svg",
        "template",
    }:
        plt.show()
    plt.close(figure)
    return destination


def display_shap_image(
    input_data: Any,
    attribution_map: Any,
    output_path: str | Path,
    title: str,
) -> Path:
    import matplotlib
    import matplotlib.pyplot as plt

    image = to_numpy(input_data)
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 2:
        image = np.repeat(image[..., np.newaxis], 3, axis=-1)
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(
            f"Expected a single image input, received shape {image.shape}."
        )

    image = np.nan_to_num(image.astype(np.float32, copy=False))
    minimum = float(image.min())
    maximum = float(image.max())
    if minimum < 0 or maximum > 1:
        image_range = maximum - minimum
        image = (
            (image - minimum) / image_range
            if image_range > 0
            else np.zeros_like(image)
        )

    attribution = to_numpy(attribution_map).squeeze()
    if attribution.ndim != 2 or attribution.shape != image.shape[:2]:
        raise ValueError(
            "Expected a 2D SHAP attribution map matching the image "
            f"dimensions; received {attribution.shape} for {image.shape[:2]}."
        )
    attribution = np.nan_to_num(attribution.astype(np.float32, copy=False))
    limit = float(np.max(np.abs(attribution)))
    if limit == 0:
        raise ValueError("The SHAP image attribution map contains only zeros.")

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots()
    axis.imshow(image)
    overlay = axis.imshow(
        attribution,
        cmap="coolwarm",
        alpha=0.45,
        vmin=-limit,
        vmax=limit,
    )
    axis.set_title(title)
    axis.axis("off")
    figure.colorbar(overlay, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    if matplotlib.get_backend().lower() not in {
        "agg",
        "cairo",
        "pdf",
        "ps",
        "svg",
        "template",
    }:
        plt.show()
    plt.close(figure)
    return destination


def display_tabular_attributions(
    feature_ranking: list[dict[str, Any]],
    output_path: str | Path,
    title: str,
) -> Path:
    import matplotlib
    import matplotlib.pyplot as plt

    if not feature_ranking:
        raise ValueError("Tabular visualization requires feature attributions.")

    features = [str(item["feature"]) for item in feature_ranking]
    scores = np.asarray(
        [float(item["importance"]) for item in feature_ranking],
        dtype=np.float64,
    )
    if not np.isfinite(scores).all():
        raise ValueError("Feature contribution scores must be finite numbers.")

    positions = np.arange(len(features))
    colors = ["#d95f02" if score < 0 else "#1b9e77" for score in scores]
    figure_height = max(3.0, 0.4 * len(features) + 1.5)
    figure, axis = plt.subplots(figsize=(9, figure_height))
    axis.barh(positions, scores, color=colors)
    axis.set_yticks(positions, labels=features)
    axis.invert_yaxis()
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Feature contribution")
    axis.set_title(title)
    axis.grid(axis="x", linestyle=":", alpha=0.4)
    axis.set_axisbelow(True)
    figure.tight_layout()

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight")
    if matplotlib.get_backend().lower() not in {
        "agg",
        "cairo",
        "pdf",
        "ps",
        "svg",
        "template",
    }:
        plt.show()
    plt.close(figure)
    return destination
