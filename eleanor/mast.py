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
           'cone_search', 'coords_from_name', 'gaia_sources_in_tpf']

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
    """Completes a cone search in the Gaia DR2 or TIC catalog.

    Parameters
    ----------
    r : float
        Radius of cone search [deg]
    service : str
        MAST service to use. Either 'Mast.Catalogs.GaiaDR2.Cone'
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
    """Crossmatches [RA,Dec] position to a source in the Gaia DR2 or TIC catalog.

    Parameters
    ----------
    pos : tuple
        (RA, Dec)
    r :  float
        Radius of search for crossmatch.
    service : str
        Name of service to use. 'Mast.GaiaDR2.Crossmatch'
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
    if isinstance(tic, int):
        tic = str(tic)
    if not tic.startswith('tic'):
        tic = 'tic' + str(tic)
    ticData = Catalogs.query_object(tic, radius=.0001, catalog="TIC")
    return [ticData['ra'].data[0], ticData['dec'].data[0]], [ticData['Tmag'].data[0]], int(ticData['version'].data[0]), ticData['contratio'].data[0]

def coords_from_name(name):
    """ Finds the RA, Dec for a given target name."""
    from astroquery.simbad import Simbad
    result_table = Simbad.query_object(name)
    coords = SkyCoord(Angle(result_table['RA'][0] + ' hours'),
                      Angle(result_table['DEC'][0] + ' degrees'))
    return (coords.ra.deg, coords.dec.deg)

def coords_from_gaia(gaia_id):
    """Returns table of Gaia DR2 data given a source_id."""
    from astroquery.gaia import Gaia
    import warnings
    warnings.filterwarnings('ignore', module='astropy.io.votable.tree')
    adql = 'SELECT gaia.source_id, ra, dec FROM gaiadr2.gaia_source AS gaia WHERE gaia.source_id={0}'.format(gaia_id)
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
    gaia = crossmatch_by_position(coords, 0.01, 'Mast.GaiaDR2.Crossmatch')
    gaiaPos = [gaia['MatchRA'], gaia['MatchDEC']]
    sepGaia = crossmatch_distance(coords, gaiaPos)
    return int(gaia[sepGaia==np.min(sepGaia)]['MatchID'].data[0])

def crossmatch_distance(pos, match):
    """Returns distance in arcsec between two sets of coordinates."""
    c1 = SkyCoord(pos[0]*u.deg, pos[1]*u.deg, frame='icrs')
    c2 = SkyCoord(match[0]*u.deg, match[1]*u.deg, frame='icrs')
    return c1.separation(c2).to(u.arcsec)

def tic_by_contamination(pos, r, contam, tmag_lim, call_internal=False):
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

def gaia_sources_in_tpf(tpf, magnitude_limit=18, dims=None):
    """
    Gets all the Gaia sources from a TESS TargetPixelFile or an eleanor.Source
    """
    if hasattr(tpf, "ra"): # it's a 'TessTargetPixelFile'
        ra, dec = tpf.ra, tpf.dec
    else: # it's a 'Source': need to specify 'dims'
        ra, dec = tpf.coords
        h, w = dims
    wcs = tpf.wcs
    c1 = SkyCoord(ra, dec, frame='icrs', unit='deg')
    # Use pixel scale for query size
    pix_scale = 21.0
    # We are querying with a diameter as the radius, overfilling by 2x.
    Vizier.ROW_LIMIT = -1
    result = Vizier.query_region(c1, catalog=["I/345/gaia2"], radius=Angle(np.max(dims) * pix_scale, "arcsec"))
    no_targets_found_message = ValueError('Either no sources were found in the query region '
                                            'or Vizier is unavailable')
    too_few_found_message = ValueError('No sources found brighter than {:0.1f}'.format(magnitude_limit))
    if result is None:
        raise no_targets_found_message
    elif len(result) == 0:
        raise too_few_found_message
    result = result["I/345/gaia2"].to_pandas()
    result = result[result.Gmag < magnitude_limit]
    if len(result) == 0:
        raise no_targets_found_message
    radecs = np.vstack([result['RA_ICRS'], result['DE_ICRS']]).T
    coords = wcs.all_world2pix(radecs, 0)
    # coords_x and coords_y have their zero at the center of the TPF
    result["coords_x"] = coords[:,0] - w/2
    result["coords_y"] = coords[:,1] - h/2
    result = result[(np.abs(result.coords_x) <= w/2) & (np.abs(result.coords_y) <= h/2)]
    return result
