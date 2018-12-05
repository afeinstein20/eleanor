from ..mast import *
from eleanor.mast import crossmatch_distance, tic_by_contamination

import astropy.units as u
from numpy.testing import assert_almost_equal

pos1 = [19, 50]
pos2 = [20, 49]

def test_contamination():
    # there should only be 1 target with contamination < 5.5 in this region
    assert(len(tic_by_contamination(pos1, 1, [0,1], 5.5)) == 1)

def test_crossmatch():
    # is distance between cooordinates calculated correctly?
    assert_almost_equal(crossmatch_distance(pos1, pos2).value, 4292.5, decimal=1)

def test_cone_search():
    # there should be 785 targets in within .01 degrees of pos1
    assert(len(cone_search(pos1, .01, 'Mast.Catalogs.GaiaDR2.Cone')) == 13)
