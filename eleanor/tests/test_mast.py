from ..mast import *
from eleanor.mast import (crossmatch_distance, tic_by_contamination,
crossmatch_by_position)

import astropy.units as u
from numpy.testing import assert_almost_equal

pos1 = [19, 50]
pos2 = [20, 49]

def test_crossmatch():
    # is distance between cooordinates calculated correctly?
    assert_almost_equal(crossmatch_distance(pos1, pos2).value, 4292.5, decimal=1)
    # assert(len(crossmatch_by_position(pos1, .01, 'Mast.GaiaDR2.Crossmatch')) == 2)

def test_cone_search():
    # there should be 13 targets in within .01 degrees of pos1
    assert(len(cone_search(pos1, .01, 'Mast.Catalogs.GaiaDR2.Cone')) == 13)

def test_coords():
    # fetch coords from Gaia, make sure they're close
    gaia_id = 403326707889616896
    tic_id = 29987116
    result = coords_from_gaia(gaia_id)
    ra, dec = result[0], result[1]
    assert_almost_equal(ra, 19, decimal=1)
    assert_almost_equal(dec, 50, decimal=1)

    '''
    # now check to make sure these coords point to the correct tic
    assert(int(tic_from_coords((ra, dec))[0]) in [tic_id, 25155310])
    assert(int(gaia_from_coords((ra, dec))) in [gaia_id, 4661584398787754240])
    '''
    # ensure coords have correct dimensions
    assert(len(coords_from_tic(tic_id)[0]) == 2)

def test_contamination():
    # there should only be 1 target with contamination < 5.5 in this region
    assert(len(tic_by_contamination(pos1, 1, [0,1], [0, 7.5])) == 1)
