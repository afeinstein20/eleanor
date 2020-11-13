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
	def __init__(self, shape, col_ref, row_ref, xc, yc, nstars, fit_idx, bkg0, **kwargs):
		self.shape = shape
		self.col_ref = col_ref
		self.row_ref = row_ref
		self.xc = xc
		self.yc = yc
		self.nstars = nstars
		self.fit_idx = fit_idx
		self.bkg0 = bkg0
		self._init_grid()
		self.bounds = np.vstack((
				np.tile([0, np.infty], (self.nstars, 1)), # fluxes on each star
				np.array([
					# [-1.0, 1.0], # xshift of the star to fit
					# [-1.0, 1.0], # yshift of the star to fit
					[0, np.infty] # background average
				])
		))

	def __call__(self, *params):
		return self.evaluate(*params)

	def default_params(self, *args):
		pass

	def evaluate(self, *args):
		pass

	def _init_grid(self):
		r, c = self.row_ref, self.col_ref
		s1, s2 = self.shape
		self.y, self.x = np.mgrid[r:r+s1-1:1j*s1, c:c+s2-1:1j*s2]

	def mean(self, flux, bkg, optpars):
		return np.sum([self.evaluate(flux[j], self.xc[j], self.yc[j], optpars) for j in range(self.nstars)], axis=0) + bkg

	def get_default_par(self, d0):
		return np.concatenate((
			np.max(d0) * np.ones(self.nstars,),
			 np.array([0, 0, self.bkg0], 
			 self.get_default_optpars())
		))

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

	def evaluate(self, flux, xo, yo, params):
		"""
		Evaluate the Gaussian model
		Parameters
		----------
		flux : np.ndarray, (nstars,)
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
		psf = np.exp(-(a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2))
		psf_sum = np.sum(psf)
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

	def evaluate(self, flux, xo, yo, params):
		a, b, c, beta = params
		dx = self.x - xo
		dy = self.y - yo
		psf = np.divide(1., np.pow(1. + a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2, beta))
		psf_sum = np.sum(psf)
		return flux * psf / psf_sum
		
class Zernike(Model):
	'''
	Use the Zernike polynomials with weights given by 'weights'; the number of polynomials = the number of weights passed in.
	'''
	def __init__(self, shape, col_ref, row_ref, directory, num_params, star_coords, apply_prior=True):
		super().__init__(shape, col_ref, row_ref)
		self.cache = {}
		self.directory = directory
		self.star_coords = star_coords
		self.apply_prior = apply_prior
		self.precompute_zernike(num_params) # this can be changed later, so it's not a class parameter; 
		# it's just faster if you do this precomputation. If you change your mind you can call precompute_zernike again.

	def get_zernike_subpath(self, i):
		return os.path.join("psf_models", "zernike", "zmode_{0}_dims_{1}_{2}_center_{3}_{4}".replace(".", "p") + ".npy".format(
			i, self.x.shape[0], self.y.shape[1], *self.star_coords
		))

	def precompute_zernike(self, n_modes):
		'''
		Precompute the first 'n_modes' Zernike polynomials over the grid given by self.x/self.y,
		and saves them to directory + get_zernike_subpath(i).
		'''
		for i in range(n_modes):
			self.zernike(i)

	def zernike_radial(self, n, m, r):
		'''
		Radial component of the (n, m)th Zernike polynomial over radial coordinates r.
		Adapted from the 'hcipy' package, https://github.com/ehpor/hcipy/blob/master/hcipy/mode_basis/zernike.py. 
		TODO proper citation of hcipy if this ends up getting used
		'''
		m = abs(m)

		if ('rad', n, m) in self.cache:
			return self.cache[('rad', n, m)]

		if n == m:
			res = r**n
		elif (n - m) == 2:
			z1 = self.zernike_radial(n, n, r)
			z2 = self.zernike_radial(n - 2, n - 2, r)

			res = n * z1 - (n - 1) * z2
		else:
			p = n
			q = m + 4

			h3 = -4 * (q - 2) * (q - 3) / float((p + q - 2) * (p - q + 4))
			h2 = h3 * (p + q) * (p - q + 2) / float(4 * (q - 1)) + (q - 2)
			h1 = q * (q - 1) / 2.0 - q * h2 + h3 * (p + q + 2) * (p - q) / 8.0

			r2 = self.zernike_radial(2, 2, r)
			res = h1 * self.zernike_radial(p, q, r) + (h2 + h3 / r2) * self.zernike_radial(n, q - 2, r)

		if self.apply_prior:
			res *= np.exp(-(r - np.sqrt(self.star_coords[0] ** 2 + self.star_coords[1] ** 2) ** 2) / 100) # arbitrarily hardcoded just to see what'll happen
		self.cache[('rad', n, m)] = res
		return res

	def zernike_azimuthal(self, m, theta):
		if ('azim', m) in self.cache:
			return self.cache[('azim', m)]

		if m < 0:
			res = np.sqrt(2) * np.sin(-m * theta)
		elif m == 0:
			return 1
		else:
			res = np.sqrt(2) * np.cos(m * theta)

		self.cache[('azim', m)] = res
		return res

	def zernike(self, i):
		'''
		Evaluates the 'i'th Zernike polynomial over (self.x - xo, self.y - yo).
		Adapted from https://github.com/ehpor/hcipy/blob/master/hcipy/mode_basis/zernike.py. 
		'''
		subpath = self.get_zernike_subpath(i)
		store_path = os.path.join(self.directory, subpath)

		if os.path.exists(store_path):
			return np.load(store_path)
		n = int((np.sqrt(8 * i + 1) - 1) / 2)
		m = 2 * i - n * (n + 2)
		x = self.x - self.star_coords[0]
		y = self.y - self.star_coords[1]
		r, theta = np.hypot(x, y), np.arctan2(y, x)
		zern = np.sqrt(n + 1) * self.zernike_azimuthal(m, theta) * self.zernike_radial(n, m, r)
		if not os.path.exists(store_path):
			os.makedirs(os.path.dirname(store_path), exist_ok=True)
		
		np.save(store_path, zern)
		return zern

	def evaluate(self, flux, weights):
		'''
		Evaluates a weighted sum of Zernike polynomials.
		'''
		psf = np.zeros_like(self.x)
		for i, w in enumerate(weights):
			psf += self.zernike(i) * w
		
		psf_sum = np.sum(psf)
		return flux * psf / psf_sum

class Lygos(Model):
	'''
	Model from https://github.com/tdaylan/lygos/blob/master/lygos/main.py
	TODO figure out citation if this ends up getting used
	'''
	def __init__(self, shape, col_ref, row_ref, **kwargs):
		super().__init__(shape, col_ref, row_ref, **kwargs)
		self.num_params = 13

	def evaluate(self, flux, xo, yo, coeffs):
		x = self.x - xo
		y = self.y - yo
		terms = np.array([x, y, x * y, x ** 2, y ** 2, x ** 2 * y, x * y ** 2, x ** 3, y ** 3])
		polysum = sum(terms * coeffs[:len(terms)])
		gauss = coeffs[9] * np.exp(-coeffs[10] * x ** 2  - coeffs[11] * y ** 2)
		psf = polysum + gauss
		return flux * psf / np.sum(psf)

class MultiGaussian(Model):
	"""
	Gaussians for N stars at a time, with 
	model = sum_star flux_star * exp(-(dx, dy) * Sigma_star_inv * (dx, dy))
	"""
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		n = self.nstars
		self.bounds = np.vstack((
				self.bounds,
				np.array([
					[0, np.infty],
					[-0.5, 0.5],
					[0, np.infty]
				]), # center a, b, c
				np.tile([-0.01, 0.01], (3 * (self.nstars - 1), 1)) # drifts on a, b, c
			))

	def get_default_optpars(self):
		# a, b, c and the drifts on each
		return np.concatenate((
			np.array([1, 0.1, 1], dtype=np.float64),
			np.repeat([0.], 3 * (self.nstars - 1))
		))

	def evaluate(self, flux, xo, yo, params):
		"""
		Evaluate the Gaussian model
		Parameters
		----------
		flux : np.ndarray, (self.nstars,)
		xo, yo : scalar
			The drift on the target star.
		params : np.ndarray, (3 * self.nstars,)
			params[0 : 3] - a, b, c
			Parameters that control the rotation angle
			and the stretch along the major axis of the Gaussian,
			such that the matrix M = [a b ; b c] is positive-definite.
			
			params[3 : 3 * self.nstars]
			Parameters representing the deviation from a, b, c on the other stars,
			e.g. on the first background star, a = params[0] + params[3], b = params[1] + params[4]
		References
		----------
		https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
		"""
		psf_flux = np.zeros_like(self.x)
		for i in range(self.nstars):
			if i == self.fit_idx:
				# only consider motion of the target star for simplicity
				xs, ys = xo, yo
				a, b, c = params[:3]
			else:
				xs, ys = self.xc[i], self.yc[i]
				if i > self.fit_idx:
					drift_ind = 3 * i
				else:
					drift_ind = 3 * (i - 1)
				a, b, c = params[:3] + params[drift_ind:drift_ind+3]
			dx = self.x - xs
			dy = self.y - ys
			psf = np.exp(-(a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2))
			psf_flux += flux[i] * psf / np.sum(psf)
		
		return psf_flux
