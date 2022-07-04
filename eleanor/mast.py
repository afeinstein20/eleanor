import os, sys, re, json, time
import requests
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astroquery.vizier import Vizier
from astroquery.mast import Catalogs
from astropy.table import Table, Column, Row
from astropy.wcs import WCS


try: # Python 3.x
    from urllib.parse import quote as urlencode
    from urllib.request import urlretrieve
    import http.client as httplib
except ImportError: # Python 2.x
    from urllib import pathname2url as urlencode
    from urllib import urlretrieve
    import httplib

__all__ = ['coords_from_tic', 'gaia_from_coords', 'coords_from_gaia', 'tic_from_coords',
           'cone_search', 'coords_from_name']

def mastQuery(request):
    """Sends a request to the MAST server.

    Parameters
    ----------
    request : str
        JSON string for request.

    Returns
    ----------
    head :
        Retrieved data headers from MAST.
    content :
        Retrieved data contents from MAST.
    """
    server = 'mast.stsci.edu'

    # Grab Python Version
    version = '.'.join(map(str, sys.version_info[:3]))
        # Create Http Header Variables
    headers = {'Content-type': 'application/x-www-form-urlencoded',
               'Accept': 'text/plain',
               'User-agent': 'python-requests/'+version}
    # Encoding the request as a json string
    requestString = urlencode(json.dumps(request))
    # Opening the https cnnection
    conn = httplib.HTTPSConnection(server)
    # Making the query
    conn.request('POST', '/api/v0/invoke', 'request='+requestString, headers)

    # Getting the response
    resp = conn.getresponse()
    head = resp.getheaders()
    content = resp.read().decode('utf-8')

    # Close the https connection
    conn.close()

    return head, content

def jsonTable(jsonObj):
    """Converts JSON return type object into an astropy Table.

    Parameters
    ----------
    jsonObj :
        Output data from `mastQuery`.

    Returns
    ----------
    dataTable : astropy.table.Table
    """
    dataTable = Table()
    for col,atype in [(x['name'],x['type']) for x in jsonObj['fields']]:
        if atype=='string':
            atype='str'
        if atype=='boolean':
            atype='bool'
        if atype=='int':
            atype='float'
        dataTable[col] = np.array([x.get(col,None) for x in jsonObj['data']],dtype=atype)
    return dataTable

def cone_search(pos, r, service):
    """Completes a cone search in the Gaia DR3 or TIC catalog.

    Parameters
    ----------
    r : float
        Radius of cone search [deg]
    service : str
        MAST service to use. Either 'Mast.Catalogs.GaiaDR3.Cone'
        or 'Mast.Catalogs.Tic.Cone' are acceptable inputs.

    Returns
    ----------
    table : astropy.table.Table
        Sources found within cone of radius r.
        See the Gaia & TIC field documentation for more information
        on returned columns.
    """
    request = {'service': service,
               'params': {'ra':pos[0], 'dec':pos[1], 'radius':r},
               'format':'json', 'removecache':True}
    headers, outString = mastQuery(request)
    return jsonTable(json.loads(outString))

def crossmatch_by_position(pos, r, service):
    """Crossmatches [RA,Dec] position to a source in the Gaia DR3 or TIC catalog.

    Parameters
    ----------
    pos : tuple
        (RA, Dec)
    r :  float
        Radius of search for crossmatch.
    service : str
        Name of service to use. 'Mast.GaiaDR3.Crossmatch'
            or 'Mast.Tic.Crossmatch' are accepted.

    Returns
    -------
    table : astropy.table.Table
        Table of crossmatch results.
    """

    crossmatchInput = {'fields': [{'name':'ra' , 'type':'float'},
                                  {'name':'dec', 'type':'float'}],
                       'data': [{'ra':pos[0], 'dec':pos[1]}]}
    request = {'service':service,
               'data':crossmatchInput,
               'params': {'raColumn':'ra', 'decColumn':'dec', 'radius':r},
               'format':'json', 'removecache':True}
    headers, outString = mastQuery(request)
    return jsonTable(json.loads(outString))

def coords_from_tic(tic):
    """Finds the RA, Dec, and magnitude for a given TIC source_id.

    Returns
    -------
    coords : tuple
        (RA, Dec) position [degrees].
    tmag : float
        TESS apparent magnitude.
    """

    ticData = Catalogs.query_object('tic'+str(tic), radius=.0001, catalog="TIC")
    return [ticData['ra'].data[0], ticData['dec'].data[0]], [ticData['Tmag'].data[0]], int(ticData['version'].data[0]), ticData['contratio'].data[0]

def coords_from_name(name):
    """ Finds the RA, Dec for a given target name."""
    from astroquery.simbad import Simbad
    result_table = Simbad.query_object(name)
    coords = SkyCoord(Angle(result_table['RA'][0] + ' hours'),
                      Angle(result_table['DEC'][0] + ' degrees'))
    return (coords.ra.deg, coords.dec.deg)

def coords_from_gaia(gaia_id):
    """Returns table of Gaia DR3 data given a source_id."""
    from astroquery.gaia import Gaia
    import warnings
    warnings.filterwarnings('ignore', module='astropy.io.votable.tree')
    adql = 'SELECT gaia.source_id, ra, dec FROM gaiadr3.gaia_source AS gaia WHERE gaia.source_id={0}'.format(gaia_id)
    job = Gaia.launch_job(adql)
    table = job.get_results()
    coords = (table['ra'].data[0], table['dec'].data[0])
    return coords

def tic_from_coords(coords):
    """Returns TIC ID, Tmag, and separation of best match(es) to input coords."""
    tess = crossmatch_by_position(coords, 0.01, 'Mast.Tic.Crossmatch')
    tessPos = [tess['MatchRa'], tess['MatchDEC']]
    sepTess = crossmatch_distance(coords, tessPos)
    subTess = tess[sepTess==np.min(sepTess)]
    return int(subTess['MatchID'].data[0]), [subTess['Tmag'].data[0]], sepTess[sepTess==np.min(sepTess)]/u.arcsec, int(subTess['version'][0]), subTess['contratio'].data[0]

def gaia_from_coords(coords):
    """Returns Gaia ID of best match(es) to input coords."""
    gaia = crossmatch_by_position(coords, 0.01, 'Mast.GaiaDR3.Crossmatch')
    gaiaPos = [gaia['MatchRA'], gaia['MatchDEC']]
    sepGaia = crossmatch_distance(coords, gaiaPos)
    return int(gaia[sepGaia==np.min(sepGaia)]['MatchID'].data[0])

def crossmatch_distance(pos, match):
    """Returns distance in arcsec between two sets of coordinates."""
    c1 = SkyCoord(pos[0]*u.deg, pos[1]*u.deg, frame='icrs')
    c2 = SkyCoord(match[0]*u.deg, match[1]*u.deg, frame='icrs')
    return c1.separation(c2).to(u.arcsec)

def tic_by_contamination(pos, r, contam, tmag_lim):
    """Allows the user to perform a counts only query.

    When unsure how many results are expcted, it is best to first perform
    a counts query to avoid memory overflow.

    Parameters
    ----------
    pos : tuple
        [RA,Dec] pair to be the center of the search.
    r : float
        Radius of cone search.
    contam : tuple
        [min,max] limits on allowed contamination.
    tmag_lim :

    Returns
    ----------
    table : astropy.table.Table
        A table of source(s) in radius
    """
    request = {'service':'Mast.Catalogs.Filtered.Tic.Position.Rows',
               'format':'json',
               'params': {'columns':'r.*',
                          'filters': [{'paramName':'contratio',
                                       'values':[{'min':contam[0], 'max':contam[1]}]},
                                      {'paramName':'Tmag',
                                       'values':[{'min':tmag_lim[0], 'max':tmag_lim[1]}]}
                                      ],
                          'ra':pos[0],
                          'dec':pos[1],
                          'timeout':600,
                          'radius':r
                          }}
    print(request)
    blob = {}
    while 'fields' not in blob:
        headers, outString = mastQuery(request)
        blob = json.loads(outString)
        if 'fields' not in blob:
            print(blob)
            print("retrying...")
            time.sleep(30)
    return jsonTable(blob)
