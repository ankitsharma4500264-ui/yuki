class YukiError(Exception):
    """Base class; messages are safe to show to the user."""


class ValidationError(YukiError):
    pass


class PermissionDenied(YukiError):
    pass


class ToolArgError(YukiError):
    pass


class ToolFailure(YukiError):
    """A tool ran but could not produce a valid result."""


class InsufficientEvidence(YukiError):
    pass


class ProviderError(YukiError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class ProviderUnavailable(ProviderError):
    pass


class VectorStoreError(YukiError):
    pass


class AgentTimeout(YukiError):
    pass


class Cancelled(YukiError):
    pass
