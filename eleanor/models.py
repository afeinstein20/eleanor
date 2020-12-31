import math
import os
from astropy.io import fits as pyfits
from lightkurve.utils import channel_to_module_output
import numpy as np
import warnings
import torch
import scipy.special
from functools import reduce
from abc import ABC
from zernike import Zern
from matplotlib import pyplot as plt

# Vaneska models of Ze Vinicius

class Model(ABC):
	"""
	Base PSF-fitting model.

	Attributes
	----------
	shape : tuple
		shape of the TPF. (row_shape, col_shape)
	col_ref, row_ref : int, int
		column and row coordinates of the bottom
		left corner of the TPF
	"""
	def __init__(self, shape, col_ref, row_ref, xc, yc, bkg0, loss, **kwargs):
		self.shape = shape
		self.col_ref = col_ref
		self.row_ref = row_ref
		self.xc = xc
		self.yc = yc
		self.bkg0 = bkg0
		self._init_grid()
		self.bounds = np.vstack((
				np.tile([0, np.infty], (len(self.xc), 1)), # fluxes on each star
				np.array([
					[-2.0, 2.0], # xshift of the star to fit
					[-2.0, 2.0], # yshift of the star to fit
					[0, np.infty] # background average
				])
		))
		self.loss = loss

	def __call__(self, *params):
		return self.evaluate(*params)

	def evaluate(self, *args):
		pass

	def _init_grid(self):
		r, c = self.row_ref, self.col_ref
		s1, s2 = self.shape
		self.y, self.x = np.mgrid[r:r+s1-1:1j*s1, c:c+s2-1:1j*s2]
		self.x = torch.tensor(self.x)
		self.y = torch.tensor(self.y)

	def mean(self, flux, xshift, yshift, bkg, optpars, norm=True):
		return sum([self.evaluate(flux[j], self.xc[j]+xshift, self.yc[j]+yshift, optpars, norm) for j in range(len(self.xc))]) + bkg

	def get_default_par(self, d0):
		return np.concatenate((
			np.max(d0) * np.ones(len(self.xc),),
			np.array([0, 0, self.bkg0]), 
			self.get_default_optpars()
		))

	def fit(self, i):
		# fit the 'i'th frame
		pass # for now


class Gaussian(Model):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.bounds = np.vstack((
				self.bounds,
				np.array([
					[0, np.infty],
					[-0.5, 0.5],
					[0, np.infty],
				])
			))

	def get_default_optpars(self):
		return np.array([1, 0, 1], dtype=np.float64)

	def evaluate(self, flux, xo, yo, params, norm=True):
		"""
		Evaluate the Gaussian model
		Parameters
		----------
		flux : np.ndarray, (len(self.xc),)
		xo, yo : scalar
			Center coordinates of the Gaussian.
		a, b, c : scalar
			Parameters that control the rotation angle
			and the stretch along the major axis of the Gaussian,
			such that the matrix M = [a b ; b c] is positive-definite.
		References
		----------
		https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
		"""
		a, b, c = params
		dx = self.x - xo
		dy = self.y - yo
		psf = torch.exp(-(a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2))
		if norm:
			psf_sum = torch.sum(psf)
		else:
			psf_sum = torch.tensor(1.)
		return flux * psf / psf_sum

class Moffat(Model):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.bounds = np.vstack((
				self.bounds,
				np.array([
					[0, np.infty],
					[-0.5, 0.5],
					[0, np.infty],
					[0.0, np.infty]
				])
			))

	def get_default_optpars(self):
		return np.array([1, 0, 1, 1], dtype=np.float64) # a, b, c, beta

	def evaluate(self, flux, xo, yo, params, norm=True):
		a, b, c, beta = params
		dx = self.x - xo
		dy = self.y - yo
		psf = torch.true_divide(torch.tensor(1.), (1. + a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2) ** (beta ** 2))
		if norm:
			psf_sum = torch.sum(psf)
		else:
			psf_sum = torch.tensor(1.)
		return flux * psf / psf_sum

# from https://discuss.pytorch.org/t/modified-bessel-function-of-order-0/18609/2
# we'll always use a Bessel function of the first kind for the Airy disk, i.e. nu = 1.
class ModifiedBesselFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, nu):
        ctx._nu = nu
        ctx.save_for_backward(inp)
        return torch.from_numpy(scipy.special.iv(nu, inp.detach().numpy()))

    @staticmethod
    def backward(ctx, grad_out):
        inp, = ctx.saved_tensors
        nu = ctx._nu
        # formula is from Wikipedia
        return 0.5* grad_out *(ModifiedBesselFn.apply(inp, nu - 1.0)+ModifiedBesselFn.apply(inp, nu + 1.0)), None

modified_bessel = ModifiedBesselFn.apply

class Airy(Model):
	'''
	Airy disk model. Currently untested.
	'''
	def __init__(self, shape, col_ref, row_ref, **kwargs):
		super().__init__(shape, col_ref, row_ref, **kwargs)

	def evaluate(self, flux, xo, yo, params):
		Rn = params # Rn = R / Rz implicitly; "R normalized"
		dx = self.x - xo
		dy = self.y - yo
		r = torch.sqrt(dx ** 2 + dy ** 2)
		bessel_arg = np.pi * r / Rn
		psf = torch.pow(2 * modified_bessel(torch.tensor(1), bessel_arg) / bessel_arg, 2)
		psf_sum = torch.sum(psf)
		return flux * psf / psf_sum

class TorchZern(Zern):
	def __init__(self, n, normalise=Zern.NORM_NOLL):
		super().__init__(n, normalise)
		self.numpy_dtype = np.float32

	# almost a clone of zernike.RZern, but with Torch operations
	def Rnm(self, k, rho):
		# Horner's method, but differentiable
		return reduce(lambda c, r: c * rho + r, self.rhotab[k,:])

	def ck(self, n, m):
		if self.normalise == self.NORM_NOLL:
			if m == 0:
				return np.sqrt(n + 1.0)
			else:
				return np.sqrt(2.0 * (n + 1.0))
		else:
			return 1.0

	def angular(self, j, theta):
		m = self.mtab[j]
		if m >= 0:
			return torch.cos(m * theta)
		else:
			return torch.sin(-m * theta)

		
class Zernike(Model):
	'''
	Fit the Zernike polynomials to the PRF, possibly after a fit from one of the other models.
	'''
	def __init__(self, shape, col_ref, row_ref, xc, yc, bkg0, loss, source, zern_n=4, base_model=None):
		# note: base_model functionality is TBD
		from .prf import make_prf_from_source
		super().__init__(shape, col_ref, row_ref, xc, yc, bkg0, loss)
		self.prf = torch.tensor(make_prf_from_source(source))
		self.z = TorchZern(zern_n)
		# x, y = torch.meshgrid(torch.linspace(-1, 1, 117), torch.linspace(-1, 1, 117))
		#self.z.make_cart_grid(x, y)
		#self.zern_pars = self.z.ZZ @ self.prf.ravel() # only need this if we make cuts based on the most important modes

	def polar_coords(self, xo, yo):
		dx = self.x - xo
		dy = self.y - yo
		rho = torch.sqrt(dx ** 2 + dy ** 2)
		theta = torch.atan2(dy, dx)
		return rho, theta

	def evaluate(self, flux, xo, yo, c, params, norm=True):
		rho, theta = self.polar_coords(xo, yo)
		rho_sc = torch.true_divide(rho, c)
		psf = sum([p * torch.tensor(np.nan_to_num(self.z.radial(k, rho_sc), 0)) * self.z.angular(k, theta) for (k, p) in enumerate(params)])
		if norm:
			psf_sum = torch.sum(psf)
		else:
			psf_sum = torch.tensor(1.)
		return flux * psf / psf_sum