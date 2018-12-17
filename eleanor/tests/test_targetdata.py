import pytest
import warnings
import numpy as np

from ..source import Source
from ..targetdata import TargetData
from ..visualize import Visualize

def test_making_tpfs():
    '''Does providing an RA/Dec pair, Gaia ID, or TIC ID for the
    SAME target produce the SAME light curve?"
    '''

    star1 = Source(coords=[63.37389, -69.226789], sector='recent')
    star2 = Source(gaia=4666498154837086208)
    star3 = Source(tic=25155310)

    data1 = TargetData(star1)
    data2 = TargetData(star2)
    data3 = TargetData(star3)

    test1 = np.sum(data1.raw_flux - data2.raw_flux) # should be zero
    test2 = np.sum(data2.raw_flux - data3.raw_flux) # should be zero

    assert(test1 == test2)
    assert(test2 == 0)

'''
def test_arb_size_tpfs():
    star = Source(tic=29987116)
    data = TargetData(star, height=15, width=12)
    assert(np.shape(data.raw_flux[0] == (15,13))) # eleanor enforces oddness
'''
