import math
import os
from astropy.io import fits as pyfits
from lightkurve.utils import channel_to_module_output
import numpy as np
import tensorflow as tf

# Vaneska models of Ze Vinicius

class Model:
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
	def __init__(self, shape, col_ref, row_ref, **kwargs):
		self.shape = shape
		self.col_ref = col_ref
		self.row_ref = row_ref
		self._init_grid()

	def __call__(self, *params):
		return self.evaluate(*params)

	def _init_grid(self):
		r, c = self.row_ref, self.col_ref
		s1, s2 = self.shape
		self.y, self.x = np.mgrid[r:r+s1-1:1j*s1, c:c+s2-1:1j*s2]

class Gaussian(Model):
	def evaluate(self, flux, xo, yo, a, b, c):
		"""
		Evaluate the Gaussian model
		Parameters
		----------
		flux : tf.Variable
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
		psf = tf.exp(-(a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2))
		psf_sum = tf.reduce_sum(psf)
		return flux * psf / psf_sum


class Moffat(Model):
	def evaluate(self, flux, xo, yo, a, b, c, beta):
		dx = self.x - xo
		dy = self.y - yo
		psf = tf.divide(1., tf.pow(1. + a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2, beta))
		psf_sum = tf.reduce_sum(psf)
		return flux * psf / psf_sum

# class MoffatAltTest(Model):


class Zernike(Model):
	'''
	Use the Zernike polynomials with weights given by 'weights'; the number of polynomials = the number of weights passed in.
	'''
	def __init__(self, shape, col_ref, row_ref, directory, num_params):
		super().__init__(shape, col_ref, row_ref)
		self.cache = {}
		self.directory = directory
		self.precompute_zernike(num_params) # this can be changed later so it's not a class parameter; 
		# it's just faster if you do this precomputation. If you change your mind you can call precompute_zernike again.

	def get_zernike_subpath(self, i):
		return os.path.join("psf_models", "zernike", "zmode_{0}_dimx_{1}_dimy_{2}.npy".format(i, self.x.shape[0], self.y.shape[1]))

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
		Evaluates the 'i'th Zernike polynomial over (self.x, self.y).
		Adapted from https://github.com/ehpor/hcipy/blob/master/hcipy/mode_basis/zernike.py. 
		'''
		subpath = self.get_zernike_subpath(i)
		store_path = os.path.join(self.directory, subpath)

		if os.path.exists(store_path):
			return np.load(store_path)
		n = int((np.sqrt(8 * i + 1) - 1) / 2)
		m = 2 * i - n * (n + 2)
		x = self.x - np.median(self.x)
		y = self.y - np.median(self.y)
		r, theta = np.hypot(x, y), np.arctan2(y, x)
		zern = np.sqrt(n + 1) * self.zernike_azimuthal(m, theta) * self.zernike_radial(n, m, r)
		if not os.path.exists(store_path):
			os.makedirs(os.path.dirname(store_path), exist_ok=True)
		
		np.save(store_path, zern)
		return zern

	def evaluate(self, flux, *weights):
		'''
		Evaluates a weighted sum of Zernike polynomials.
		xo and yo are excluded, because ideally those are fitted by tip and tilt.
		'''
		psf = np.zeros_like(self.x)
		for i, w in enumerate(weights):
			psf += self.zernike(i) * w
		
		psf_sum = tf.reduce_sum(psf)
		return flux * psf / psf_sum

class Lygos(Model):
	'''
	Model from https://github.com/tdaylan/lygos/blob/master/lygos/main.py
	TODO figure out citation if this ends up getting used
	'''
	def __init__(self, shape, col_ref, row_ref, **kwargs):
		super().__init__(shape, col_ref, row_ref, **kwargs)
		self.num_params = 12

	def evaluate(self, flux, *coeffs):
		x, y = self.x, self.y
		terms = [tf.Variable(v) for v in [x, y, x * y, x ** 2, y ** 2, x ** 2 * y, x * y ** 2, x ** 3, y ** 3]]
		
		mult_coeffs, misc_coeffs = np.array(coeffs[:len(terms)]), np.array(coeffs[len(terms):])

		return (tf.reduce_sum([m * t for m, t in zip(mult_coeffs, terms)]) + misc_coeffs[0]) * tf.math.exp(-x ** 2 / misc_coeffs[1] - y ** 2 / misc_coeffs[2])
		