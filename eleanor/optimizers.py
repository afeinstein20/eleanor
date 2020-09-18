import importlib
from functools import partialmethod
import numpy as np

class OptimizerAPI:
    '''
    Common API to several optimizers, to allow for easy switching and comparisons.
    This likely won't be what this looks like in the final PR to master, and is more a utility for quick benchmarking/
    for encapsulating the eventual optimizer in case further downstream changes are desired.

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

    bounds_converter : callable
    Converts a tf var_to_bounds to a form usable by torch and numpy.

    minimize : callable
    The optimizer's minimization interface: takes in 
    - a callable objective function
    - a parameter array of type self.param
    - var_to_bounds, as returned by bounds_converter.
    - **kwargs, to handle case-by-case inputs
    '''
    def __init__(self, base_package="scipy"):
        self.pkg = importlib.import_module(base_package)
        self.float = self.pkg.float64
        self.param = eval({
            "tensorflow" : "self.pkg.Variable",
            "torch" : "partial(self.pkg.tensor, requires_grad=True)",
        }.get(base_package, "np.array"))

        self.empty = eval({
            "tensorflow" : "self.pkg.placeholder",
            "torch" : "self.pkg.empty",
        }.get(base_package, "np.empty"))

        self.math = eval({
            "tensorflow" : "self.pkg.math",
            "torch" : "self.pkg",
        }.get(base_package, "np"))

        self.concat = eval({
            "tensorflow" : "self.pkg.concat",
            "torch" : "self.pkg.cat",
        }.get(base_package, "np.concat"))

        self.bounds_converter = eval({
            "tensorflow" : "lambda var_to_bounds : var_to_bounds"
        }.get(base_package, "lambda var_to_bounds : self.pkg.stack([var_to_bounds.get(k) for k in var_to_bounds])"))

        self.minimizer = eval({
            "scipy" : "self.pkg.optimize"
        }.get(base_package, "self._minimizer"))

        self.minimize = eval({
            "torch" : "self.minimizer.step",
        }.get(base_package, "self.minimizer.minimize"))

        if base_package == "tensorflow":
            self.math.sum = self.pkg.reduce_sum

    def _minimizer(self, loss, var_list, var_to_bounds, algorithm="Adam"):
        name = self.pkg.__name__
        assert name == "torch" or name == "tensorflow"
        if name == "torch":
            pkg_name = "self.pkg.optim"
        elif name == "tensorflow":
            pkg_name = "self.pkg.keras.optimizers"
        return eval("{0}.{1}".format(pkg_name, algorithm))

    
