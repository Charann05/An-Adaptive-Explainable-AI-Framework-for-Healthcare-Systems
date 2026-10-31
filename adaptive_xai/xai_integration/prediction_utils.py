from __future__ import annotations

from typing import Any

import numpy as np


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def predicted_value(
    output: Any,
    task: str | None = None,
    *,
    class_scores: bool = False,
) -> Any:
    values = to_numpy(output)
    flattened = values.reshape(-1)
    if flattened.size == 0:
        raise ValueError("The model returned an empty prediction.")
    if task and task.lower() == "regression":
        return flattened[0].item()
    if class_scores and values.ndim == 1 and values.size > 1:
        return int(np.argmax(values))
    if values.ndim >= 2 and values.shape[-1] > 1:
        return int(np.argmax(values[0]))
    return flattened[0].item()


def probability_matrix(output: Any) -> np.ndarray:
    values = to_numpy(output)
    if values.ndim == 0:
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] == 1:
        probability = values[:, 0]
        if np.all((probability >= 0) & (probability <= 1)):
            return np.column_stack([1 - probability, probability])
        probability = 1 / (1 + np.exp(-probability))
        return np.column_stack([1 - probability, probability])
    row_sums = values.sum(axis=1)
    if np.all(values >= 0) and np.allclose(row_sums, 1, atol=1e-5):
        return values
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def predicted_class_and_confidence(output: Any) -> tuple[int, float]:
    probabilities = probability_matrix(output)
    predicted_class = int(np.argmax(probabilities[0]))
    return predicted_class, float(probabilities[0, predicted_class])
