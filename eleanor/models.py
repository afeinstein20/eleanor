import math
import os
from astropy.io import fits as pyfits
from lightkurve.utils import channel_to_module_output
import numpy as np
import warnings
from abc import ABC

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
	def __init__(self, shape, col_ref, row_ref, xc, yc, nstars, bkg0, **kwargs):
		self.shape = shape
		self.col_ref = col_ref
		self.row_ref = row_ref
		self.xc = xc
		self.yc = yc
		self.nstars = nstars
		self.bkg0 = bkg0
		self._init_grid()

	def __call__(self, *params):
		return self.evaluate(*params)

	def _init_grid(self):
		r, c = self.row_ref, self.col_ref
		s1, s2 = self.shape
		self.y, self.x = np.mgrid[r:r+s1-1:1j*s1, c:c+s2-1:1j*s2]

	def mean(self, flux, xshift, yshift, bkg, optpars):
		return np.sum([self.evaluate(flux[j], self.xc[j]+xshift, self.yc[j]+yshift, *optpars) for j in range(self.nstars)], axis=0) + bkg

	def get_default_par(self, d0):
		return np.concatenate((np.max(d0) * np.ones(self.nstars,), np.array([0, 0, self.bkg0], self.get_default_optpars())))

class Gaussian(Model):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.bounds = np.vstack((
				np.tile([0, np.infty], (self.nstars, 1)),
				np.array([
					[-1.0, 1.0],
					[-1.0, 1.0],
					[0, np.infty],
					[0, np.infty],
					[-0.5, 0.5],
					[0, np.infty],
				])
			))

	def get_default_optpars(self):
		return np.array([1, 0, 1], dtype=np.float64)

	def evaluate(self, flux, xo, yo, a, b, c):
		"""
		Evaluate the Gaussian model
		Parameters
		----------
		flux : np.ndarray, (nstars,)
		xo, yo : tf.Variable, tf.Variable
			Center coordinates of the Gaussian.
		a, b, c : tf.Variable, tf.Variable
			Parameters that control the rotation angle
			and the stretch along the major axis of the Gaussian,
			such that the matrix M = [a b ; b c] is positive-definite.
		References
		----------
		https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
		"""
		dx = self.x - xo
		dy = self.y - yo
		psf = np.exp(-(a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2))
		psf_sum = np.sum(psf)
		return flux * psf / psf_sum

	
class Moffat(Model):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.bounds = np.vstack((
				np.tile([0, np.infty], (self.nstars, 1)),
				np.array([
					[-2.0, 2.0],
					[-2.0, 2.0],
					[0, np.infty],
					[0., 3.0],
					[-0.5, 0.5],
					[0., 3.0],
				])
			))

	def get_default_optpars(self):
		return np.array([1, 0, 1, 1], dtype=np.float64)

	def evaluate(self, flux, xo, yo, a, b, c, beta):
		dx = self.x - xo
		dy = self.y - yo
		psf = np.divide(1., np.pow(1. + a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2, beta))
		psf_sum = np.sum(psf)
		return flux * psf / psf_sum
		