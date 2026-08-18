from .exceptions import PluginNotFoundError


_PLUGIN_REGISTRY = {}


def register_plugin(name: str):
    """
    Register an XAI plugin using its technique name.
    """

    technique = name.strip()

    if not technique:
        raise ValueError("Plugin name cannot be empty.")

    def decorator(plugin_class):

        if technique in _PLUGIN_REGISTRY:
            raise RuntimeError(
                f"Plugin already registered: {technique}"
            )

        _PLUGIN_REGISTRY[technique] = plugin_class()

        return plugin_class

    return decorator


def get_plugin(name: str):
    """
    Retrieve a registered plugin instance.
    """

    technique = name.strip()

    try:
        return _PLUGIN_REGISTRY[technique]

    except KeyError:
        raise PluginNotFoundError(
            f"No XAI plugin registered for: {technique}"
        )


def get_registered_plugins() -> list[str]:
    """
    Return all registered XAI techniques.
    """

    return sorted(_PLUGIN_REGISTRY.keys())