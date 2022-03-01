__all__ = ['EleanorWarning', 'SearchError', 'EdgeProblem']

class EleanorWarning(Warning):
    """A class to hold Eleanor-specific warnings."""
    pass

class SearchError(Exception):
    """Exception raised when no target was found."""
    pass

class EdgeProblem(Exception):
    """ Exception raised when target is on the edge
        of the detector. """
    pass
