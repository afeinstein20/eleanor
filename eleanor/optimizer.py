import importlib
from functools import partial
import numpy as np
import torch
import tqdm

class OptimizerAPI:
    """
    This provides an optimizer API, to separate targetdata.py from the optimizer details.

    Attributes
    ----------
    variables : iterable
    List of variable objects (torch.tensor) to be optimized.

    bounds : np.ndarray
    A dictionary from variable names to (lower, upper) tuples for each variable.

    loss : callable
    Takes in an model data array and returns its numerical loss value; function to be minimized.

    data : list or np.ndarray or torch.tensor
    The target data to compare the model data array to; one argument to the loss.

    num_opt_steps : int
    The number of steps in an optimization.
    """
    def __init__(self, model, variables=None, bounds=None, likelihood=None, data=None, **kwargs):
        self.model = model
        self.likelihood = likelihood
        if variables is not None:
            self.set_variables(variables)
        if bounds is not None:
            self.set_bounds(bounds)
        if data is not None:
            self.set_data_and_loss(likelihood, *data)
        self.num_opt_steps = 50
        for k in kwargs:
            setattr(self, k, kwargs.get(k))

    def __repr__(self):
        # change this as desired, if the optimization package ever changes.
        return "Interface for optimizer based on the PyTorch package."

    def set_variables(self, variables):
        if isinstance(variables, torch.Tensor):
            self.variables = [variables]
        else:
            self.variables = []
            for v in variables:
                if isinstance(v, torch.Tensor):
                    self.variables.append(v)
                else:
                    self.variables.append(torch.tensor(v, dtype=torch.float64, requires_grad=True))

    def set_bounds(self, bounds):
        self.bounds = torch.tensor(bounds)

    def set_data_and_loss(self, flux_raw, flux_err, bkg):
        self.flux_raw = torch.tensor(flux_raw.astype(np.float32), dtype=torch.float64)
        self.flux_err = torch.tensor(flux_err.astype(np.float32), dtype=torch.float64)
        self.bkg = torch.tensor(bkg.astype(np.float32), dtype=torch.float64)
        if self.likelihood == 'gaussian':
            self.loss = partial(torch.nn.MSELoss(reduction='sum'), self.flux_raw)
        elif self.likelihood == 'poisson':
            self.loss = lambda mean: torch.sum(torch.subtract(mean+self.bkg, torch.multiply(self.flux_raw+self.bkg, torch.log(mean+self.bkg))))
        else:
            raise ValueError("likelihood argument {0} not supported".format(likelihood))
    
    def minimize(self, algorithm="Adam"):
        if any([x is None for x in [self.variables, self.bounds, self.loss]]):
            raise ValueError("Set variables, bounds, and loss before optimizing.")
        opt = eval("torch.optim.{}".format(algorithm))(params=self.variables, lr=0.1)
        for i in range(self.num_opt_steps):
            loss = self.loss(self.model.get_mean(self.variables))
            opt.zero_grad()
            loss.backward()
            opt.step()
        return self.model.get_mean(self.variables)
