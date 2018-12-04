__all__ = ['EleanorWarning', 'SearchError']

class EleanorWarning(Warning):
    """A class to hold Eleanor-specific warnings."""
    pass

class SearchError(Exception):
    """Exception raised when no target was found."""
    pass
