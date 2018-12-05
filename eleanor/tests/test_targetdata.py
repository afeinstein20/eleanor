import pytest
import warnings

from ..source import Source
from ..targetdata import TargetData
import matplotlib.pyplot as plt
from ..visualize import Visualize

def test_making_tpfs():
    '''Does providing an RA/Dec pair, Gaia ID, or TIC ID for the
    SAME target produce the SAME light curve?"
    '''

    star1 = Source(coords=(84.2917, -80.4689))
    star2 = Source(gaia=4623036865373793408)
    star3 = Source(tic=261136679)

    data1 = TargetData(star1)
    data2 = TargetData(star2)
    data3 = TargetData(star3)

    test1 = np.sum(data1.tpf.raw_flux - data2.tpf.raw_flux) # should be zero
    test2 = np.sum(data2.tpf.raw_flux - data3.tpf.raw_flux) # should be zero

    assert(test1 == test2)
    assert(test2 == 0)

def test_arb_size_tpfs():
    star = Source(tic=261136246)
    data = TargetData(star, height=15, width=12)
    assert(np.shape(data.tpf.raw_flux[0] == (15,12)))



test_making_tpfs()
