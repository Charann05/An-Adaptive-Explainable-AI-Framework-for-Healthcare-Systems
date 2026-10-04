from typing import Type

from .base_adapter import ModelAdapter


_ADAPTER_REGISTRY: list[Type[ModelAdapter]] = []


def register_adapter(adapter_class: Type[ModelAdapter]):
    """
    Register a model-framework adapter.
    """

    if adapter_class in _ADAPTER_REGISTRY:
        return adapter_class

    _ADAPTER_REGISTRY.append(adapter_class)

    return adapter_class


def resolve_adapter(
    model,
    model_info: dict,
) -> ModelAdapter:
    """
    Find an adapter capable of handling the supplied model.
    """

    for adapter_class in _ADAPTER_REGISTRY:

        adapter = adapter_class()

        try:
            if adapter.can_handle(
                model,
                model_info,
            ):
                return adapter

        except Exception:
            continue

    framework = model_info.get(
        "framework",
        "unknown",
    )

    raise ValueError(
        "No compatible model adapter found. "
        f"Detected framework: {framework}"
    )


def get_registered_adapters() -> list[str]:
    return [
        adapter.framework
        for adapter in _ADAPTER_REGISTRY
    ]