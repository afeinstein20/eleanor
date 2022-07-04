import math
import tensorflow as tf
from astropy.io import fits as pyfits
from lightkurve.utils import channel_to_module_output
import numpy as np

# Vaneska models of Ze Vinicius

class Model:
    """
    Pretty dumb Gaussian model.
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

    def _init_grid(self):
        r, c = self.row_ref, self.col_ref
        s1, s2 = self.shape
        self.y, self.x = np.mgrid[r:r+s1-1:1j*s1, c:c+s2-1:1j*s2]


class Gaussian(Model):
    def __call__(self, *params):
        return self.evaluate(*params)

    def evaluate(self, flux, xo, yo, a, b, c):
        """
        Evaluate the Gaussian model
        Parameters
        ----------
        flux : tf.Variable
        xo, yo : tf.Variable, tf.Variable
            Center coordiantes of the Gaussian.
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
    def __call__(self, *params):
        return self.evaluate(*params)

    def evaluate(self, flux, xo, yo, a, b, c, beta):
        dx = self.x - xo
        dy = self.y - yo
        psf = tf.divide(1., tf.pow(1. + a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2, beta))
        psf_sum = tf.reduce_sum(psf)
        return flux * psf / psf_sum
