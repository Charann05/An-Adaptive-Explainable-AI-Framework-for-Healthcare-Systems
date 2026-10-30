class XAIIntegrationError(Exception):
    """Base exception for the XAI integration layer."""


class PluginNotFoundError(XAIIntegrationError):
    """Requested XAI plugin is not registered."""


class PluginCompatibilityError(XAIIntegrationError):
    """Plugin cannot handle the supplied model."""


class DependencyError(XAIIntegrationError):
    """Required XAI dependency could not be prepared."""


class ModelLoadError(XAIIntegrationError):
    """Model could not be loaded."""


class ModelCompatibilityError(ModelLoadError):
    """Model data is incompatible with the installed framework version."""

    def __init__(
        self,
        message: str,
        *,
        package: str | None = None,
        saved_versions: tuple[str, ...] = (),
        installed_version: str | None = None,
    ) -> None:
        super().__init__(message)
        self.package = package
        self.saved_versions = saved_versions
        self.installed_version = installed_version


class ExplanationExecutionError(XAIIntegrationError):
    """XAI explanation execution failed."""