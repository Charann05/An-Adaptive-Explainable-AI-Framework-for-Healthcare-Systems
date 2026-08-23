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


class ExplanationExecutionError(XAIIntegrationError):
    """XAI explanation execution failed."""