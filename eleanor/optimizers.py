import importlib
from functools import partial
import numpy as np

class OptimizerAPI:
    '''
    Common API to several optimizers, to allow for easy switching and comparisons.

    Attributes
    ----------
    pkg : module
    The package being wrapped.

    param : callable
    The method to get an optimization parameter variable.
    Must take in a value and dtype.

    float : dtype
    The optimizer's float type.

    empty : callable
    The optimizer's "empty array/tensor" function.

    math : module
    The optimizer's math module.

    concat : callable
    The optimizer's concatenation function.

    minimize : callable
    The optimizer's minimization interface: takes in 
    - a callable objective function
    - a parameter array of type self.param
    - 
    '''
    def __init__(self, base_package="tf", **kwargs):
        self.pkg = importlib.import_module(base_package)
        self.float = self.pkg.float64
        self.param = eval({
            "tf" : "self.pkg.Variable",
            "torch" : "partial(self.pkg.tensor, requires_grad=True)",
        }.get(base_package, "np.array"))

        self.empty = eval({
            "tf" : "self.pkg.placeholder",
            "torch" : "self.pkg.empty",
        }.get(base_package, "np.empty"))

        self.math = eval({
            "tf" : "self.pkg.math",
            "torch" : "self.pkg",
        }.get(base_package, "np"))

        self.concat = eval({
            "tf" : "self.pkg.concat",
            "torch" : "self.pkg.cat",
        }).get(base_package, "np.concat")

        self.minimize = eval({
            "tf" : "partial(self.pkg.contrib.opt.ScipyOptimizerInterface, )",
            "torch" : 
        })

        if base_package == "tf":
            self.math.sum = self.pkg.reduce_sum

        for k in kwargs:
            setattr(self, k, kwargs[k])

    
