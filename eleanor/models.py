import math

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
    def __init__(self, shape, col_ref, row_ref):
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

class Zernike(Model):
    '''
    Use the Zernike polynomials with weights given by 'weights'; the number of polynomials = the number of weights passed in.
    question for myself for later: what's the preferred datapath for caching these results? ~/.eleanor/...
    '''
    def zernike_radial(n, m, r):
        '''
        Radial component of the (n, m)th Zernike polynomial over radial coordinates r.
        Adapted from the 'hcipy' package, https://github.com/ehpor/hcipy/blob/master/hcipy/mode_basis/zernike.py. 
        TODO proper citation of hcipy if this ends up getting used
        '''
        m = abs(m)

        if n == m:
            res = r**n
        elif (n - m) == 2:
            z1 = zernike_radial(n, n, r, cache)
            z2 = zernike_radial(n - 2, n - 2, r, cache)

            res = n * z1 - (n - 1) * z2
        else:
            p = n
            q = m + 4

            h3 = -4 * (q - 2) * (q - 3) / float((p + q - 2) * (p - q + 4))
            h2 = h3 * (p + q) * (p - q + 2) / float(4 * (q - 1)) + (q - 2)
            h1 = q * (q - 1) / 2.0 - q * h2 + h3 * (p + q + 2) * (p - q) / 8.0

            r2 = zernike_radial(2, 2, r, cache)
            res = h1 * zernike_radial(p, q, r, cache) + (h2 + h3 / r2) * zernike_radial(n, q - 2, r, cache)

        return res

    def zernike_azimuthal(m, theta):
        if m < 0:
            res = np.sqrt(2) * np.sin(-m * theta)
        elif m == 0:
            return 1
        else:
            res = np.sqrt(2) * np.cos(m * theta)

        return res

    def zernike(i):
        '''
        Evaluates the 'i'th Zernike polynomial over (self.x, self.y).
        Adapted from https://github.com/ehpor/hcipy/blob/master/hcipy/mode_basis/zernike.py. 
        '''
        n = int((np.sqrt(8 * i + 1) - 1) / 2)
        r, theta = tf.math.sqrt(self.x ** 2 + self.y ** 2), tf.math.tan(self.y / self.x)
        return np.sqrt(n + 1) * zernike_azimuthal(m, theta) * zernike_radial(n, m, r)

    def evaluate(self, flux, *weights):
        '''
        Evaluates a weighted sum of Zernike polynomials.
        xo and yo are excluded, because ideally those are fitted by tip and tilt.
        '''
        psf = np.zeros_like(self.x)
        for i, w in enumerate(weights):
            psf += zernike(i) * w
        
        psf_sum = tf.reduce_sum(psf)
        return flux * psf / psf_sum
