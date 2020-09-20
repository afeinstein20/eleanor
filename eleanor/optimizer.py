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

    bounds : dict
    A dictionary from variable names to (lower, upper) tuples for each variable.

    loss : callable
    Takes in an model data array and returns its numerical loss value; function to be minimized.

    data : list or np.ndarray or torch.tensor
    The target data to compare the model data array to; one argument to the loss.

    num_opt_steps : int
    The number of steps in an optimization.
    """
    def __init__(self, model, variables=None, bounds=None, loss_name=None, data=None, **kwargs):
        self.model = model
        if variables is not None:
            self.set_variables(variables)
        if bounds is not None:
            self.set_bounds(bounds)
        if data is not None:
            self.set_data_and_bkg(*data)
        self.loss_name = loss_name
        self.num_opt_steps = 1000
        for k in kwargs:
            setattr(self, k, kwargs.get(k))

    def __repr__(self):
        # change this as desired, if the optimization package ever changes.
        return "Interface for optimizer based on the PyTorch package."

    def set_variables(self, variables):
        self.variables = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in variables]

    def set_bounds(self, bounds):
        self.bounds = torch.tensor(bounds)

    def set_data_and_bkg(self, flux_raw, flux_err, bkg):
        self.flux_raw = torch.tensor(flux_raw)
        self.flux_err = torch.tensor(flux_err)
        self.bkg = torch.tensor(bkg)
        self.set_loss(self.loss_name)

    def set_loss(self, loss_name):
        if loss_name == 'gaussian':
            self.loss = partial(torch.nn.MSELoss(reduction='sum'), self.flux_raw)
        elif loss_name == 'poisson':
            self.loss = lambda mean: torch.sum(torch.subtract(mean+self.bkg, torch.multiply(self.flux_raw+self.bkg, optimizer.math.log(mean+self.bkg))))
        else:
            raise ValueError("likelihood argument {0} not supported".format(loss_name))
    
    def minimize(self, algorithm="SGD"):
        if any([x is None for x in [self.variables, self.bounds, self.loss]]):
            raise ValueError("Set variables, bounds, and loss before optimizing.")
        opt = eval("torch.optim.{}".format(algorithm))(params=self.variables, lr=0.001)
        for _ in range(self.num_opt_steps):
            self.model.set_mean(self.variables)
            loss = self.loss(self.model.mean)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.model.set_mean(self.variables)
        # return self.variables



