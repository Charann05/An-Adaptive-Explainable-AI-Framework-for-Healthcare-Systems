from pathlib import Path
from typing import Any

import numpy as np
import matplotlib

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
