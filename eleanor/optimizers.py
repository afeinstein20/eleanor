import importlib
from functools import partial
import numpy as np

class Optimizer:
    '''
    Common API to several optimizers, to allow for easy switching and comparisons.

    Attributes
    ----------
    pkg : module
    The package being wrapped.

    param : callable
    The method to get an optimization parameter variable.
    Must take in a value and dtype.
    '''
    def __init__(self, base_package="tf", **kwargs):
        self.pkg = importlib.import_module(base_package)
        self.float = self.pkg.float64
        self.param = eval({
            "tf" : "self.pkg.Variable",
            "torch" : "partial(self.pkg.tensor, requires_grad=True)",
            "scipy" : "np.array"
        }.get(base_package))
        self.empty = eval({
            "tf" : "self.pkg.placeholder",
            "torch" : "self.pkg.empty",
            "scipy" : "np.empty"
        }.get(base_package))

        for k in kwargs:
            setattr(self, k, kwargs[k])

    
