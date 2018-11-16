import os, sys, re, json, time
import requests
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
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
           'cone_search']

def mastQuery(request):
    """
    Sends a request to the MAST server
    Parameters
    ----------
        request: json string
    Returns
    ----------
        head: headers for response
        content: data for response
    """
    t0 = time.time()
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
    print(time.time()-t0)

    return head, content


def jsonTable(jsonObj):
    """
    Convets json return type object into an astropy Table
    Parameters
    ----------
        jsonObj: an object from mastQuery
    Returns
    ----------
        table: astropy table for jsonObj
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


def cone_search(pos, r, service, multiPos=None):
    """
    Completes a cone search in the Gaia DR2 or TIC catalog
    Parameters
    ----------
        r: radius of cone search [deg]
        service: identifies which MAST service to use. Either 'Mast.Catalogs.GaiaDR2.Cone'
                 or 'Mast.Catalogs.Tic.Cone' are acceptable inputs
    Returns
    ----------
        table: table of sources found within cone of radius r
        See the Gaia & TIC field documantation for more information on returned columns
    """
    if multiPos != None:
        pos = multiPos
    else:
        pos = pos
    request = {'service': service,
               'params': {'ra':pos[0], 'dec':pos[1], 'radius':r},
               'format':'json'}
    headers, outString = mastQuery(request)
    return jsonTable(json.loads(outString))


def crossmatch_by_position(pos, r, service):
    """
    Crossmatches [RA,Dec] position to a source in the Gaia DR2 catalog and TIC catalog
    Parameters
    ----------
        pos: [RA,Dec] list
        r: radius of search for crossmatch
        service: identifies which catalog to crossmatch to. Either: 'Mast.GaiaDR2.Crossmatch'
                 or 'Mast.Tic.Crossmatch' are accepted
        multiPos: used when user passes in list of IDs to crossmatch
    Returns
    ----------
        gaiaTable: table of crossmatch results in Gaia DR2
        ticTable : table of crossmatch results in TIC
    """

    crossmatchInput = {'fields': [{'name':'ra' , 'type':'float'},
                                  {'name':'dec', 'type':'float'}],
                       'data': [{'ra':pos[0], 'dec':pos[1]}]}
    request = {'service':service,
               'data':crossmatchInput,
               'params': {'raColumn':'ra', 'decColumn':'dec', 'radius':r},
               'format':'json'}
    headers, outString = mastQuery(request)
    return jsonTable(json.loads(outString))


def gaia_pos_by_ID(gaiaID, multiSource=None):
    """
    Finds the RA,Dec for a given Gaia source_id
    Parameters
    ----------
    Returns
    ----------
        source_id, pos [RA,Dec], gmag, pmra, pmdec, parallax
    """
    from astroquery.gaia import Gaia

    if multiSource != None:
        source = multiSource
    else:
        source = gaia

    adql = 'SELECT gaia.source_id, gaia.ra, gaia.dec, gaia.phot_g_mean_mag, gaia.pmra, gaia.pmdec, gaia.parallax FROM gaiadr2.gaia_source AS gaia WHERE gaia.source_id={0}'.format(source)

    job = Gaia.launch_job(adql)
    table = job.get_results()
    return table


def coords_from_tic(tic, multiSource=None):
    """
    Finds the RA,Dec for a given TIC source_id
    Parameters
    ----------
        multiSource: used when user passes in a file of TIC IDs to crossmatch
    Returns
    ----------
        source_id, pos [RA,Dec], tmag
    """
    if multiSource != None:
        source = multiSource
    else:
        source = tic
    ticData = Catalogs.query_criteria(catalog='Tic', ID=[source, source])
    return [ticData['ra'].data[0], ticData['dec'].data[0]], ticData['Tmag'].data

def coords_from_gaia(gaia_id):
    """ Grabs coordinates of input Gaia ID from Gaia DR2 Catalog """
    from astroquery.gaia import Gaia
    adql = 'SELECT gaia.source_id FROM gaiadr2.gaia_source AS gaia WHERE gaia.source_id={0}'.format(gaia_id)
    job = Gaia.launch_job(adql)
    table = job.get_results()
    return table


def tic_from_coords(coords):
    """ Grabs coordinates of input TIC ID from Tess Input Catalog v7 """
    tess = crossmatch_by_position(coords, 0.5, 'Mast.Tic.Crossmatch')[0]
    tessPos = [tess['MatchRA'], tess['MatchDEC']]
    sepTess = crossmatch_distance(coords, tessPos)
    return tess['MatchID'], [tess['Tmag']], sepTess/u.arcsec


def gaia_from_coords(coords):
    gaia = crossmatch_by_position(coords, 0.01, 'Mast.GaiaDR2.Crossmatch')[0]
    gaiaPos = [gaia['MatchRA'], gaia['MatchDEC']]
    return gaia['MatchID']



def initialize_table():
    """ Creates a table for crossmatching multiple sources between Gaia and TIC catalogs """
    columns = ['Gaia_ID', 'TIC_ID', 'RA', 'Dec', 'separation', 'Gmag', 'Tmag', 'pmra', 'pmdec', 'parallax']
    t = Table(np.zeros(10), names=columns)
    t['RA'].unit, t['Dec'].unit, t['separation'].unit = u.arcsec, u.arcsec, u.arcsec
    t['pmra'].unit, t['pmdec'].unit = u.mas/u.year, u.mas/u.year
    return t

def crossmatch_distance(pos, match):
    """ Finds distance between source and crossmatched source(s) """
    c1 = SkyCoord(pos[0]*u.deg, pos[1]*u.deg, frame='icrs')
    c2 = SkyCoord(match[0]*u.deg, match[1]*u.deg, frame='icrs')
    return c1.separation(c2).to(u.arcsec)


def crossmatch_multi_to_gaia(fn, r=0.01):
    """
    Crossmatches file of TIC IDs to Gaia
    Parameters
    ----------
        fn: file of TIC IDs
        r: radius of cone search. Defaults to r=0.01
    Returns
    ----------
        table: table of gaia_id, tic_id, ra, dec, delta_pos, gmag, tmag, pmra, pmdec, parallax
    """
    sources  = np.loadtxt(fn, dtype=int)
    service  = 'Mast.GaiaDR2.Crossmatch'
    r = 0.01
    t = initialize_table()
    for s in sources:
        tID, pos, tmag = tic_pos_by_ID(s)
        gaia = crossmatch_by_position(pos, r, service)
        pos[0], pos[1] = pos[0]*u.deg, pos[1]*u.deg
        separation = crossmatch_distance(pos, [gaia['MatchRA'], gaia['MatchDEC']])
        t.add_row([gaia['MatchID'], s, (pos[0]).to(u.arcsec), (pos[1]).to(u.arcsec), separation, gaia['phot_g_mean_mag'], tmag, gaia['pmra'], gaia['pmdec'], gaia['parallax']])
    t.remove_row(0)
    return t


def crossmatch_multi_to_tic(fn, r=0.1):
    """
    Crossmatches file of Gaia IDs to TIC
    Parameters
    ----------
        fn: file of Gaia IDs
        r:  radius of cone search. Defaults to r=0.1
    Returns
    ----------
        table: table of gaia_id, tic_id, ra, dec, delta_pos, gmag, tmag, pmra, pmdec, parallax
    """
    sources = np.loadtxt(list, dtype=int)
    service  = 'Mast.Tic.Crossmatch'
    t = initialize_table()
    for s in sources:
        table = gaia_pos_by_ID(s)
        sID, pos, gmag = table['source_id'].data, [table['ra'].data[0], table['dec'].data[0]], table['phot_g_mean_mag'].data
        pmra, pmdec, parallax = table['pmra'].data, table['pmdec'].data, table['parallax'].data
        tic = crossmatch_by_position(pos, r, service)
        pos = [pos[0]*u.deg, pos[1]*u.deg]
        separation = crossmatch_distance(pos, [tic['MatchRA'], tic['MatchDEC']])
        min = separation.argmin()
        row = tic[min]
        t.add_row([sID, row['MatchID'], (pos[0]).to(u.arcsec), (pos[1]).to(u.arcsec), separation[min], gmag, row['Tmag'], pmra, pmdec, parallax])
    t.remove_row(0)

    return t


def find_by_position(pos):
    """
    Allows the user to pass in a file of RA,Dec pairs to be matched in Gaia & TIC
    Parameters
    ----------
    Returns
    ----------
        table: table of gaia_id, tic_id, ra, dec, delta_pos_gaia, delta_pos_tic, gmag, tmag, pmra, pmdec, parallax
    """
    columns = ['Gaia_ID', 'TIC_ID', 'RA', 'Dec', 'Gaia_sep', 'TIC_sep', 'Gmag', 'Tmag', 'pmra', 'pmdec', 'parallax']
    t = Table(np.zeros(11), names=columns)
    t['RA'].unit, t['Dec'].unit, t['Gaia_sep'].unit, t['TIC_sep'].unit = u.arcsec, u.arcsec, u.arcsec, u.arcsec
    t['pmra'].unit, t['pmdec'].unit = u.mas/u.year, u.mas/u.year

    for i in range(len(data)):
        pos = data[i]
        gaia = crossmatch_by_position(pos, 0.01, 'Mast.GaiaDR2.Crossmatch', pos)[0]
        tess = crossmatch_by_position(pos, 0.5, 'Mast.Tic.Crossmatch', pos)[0]
        pos[0], pos[1] = pos[0]*u.deg, pos[1]*u.deg
        gaiaPos = [gaia['MatchRA'], gaia['MatchDEC']]
        sepGaia = crossmatch_distance(pos, gaiaPos)
        tessPos = [tess['MatchRA'], tess['MatchDEC']]
        sepTess = crossmatch_distance(pos, tessPos)

        t.add_row([gaia['MatchID'], tess['MatchID'], pos[0], pos[1], sepGaia, sepTess, gaia['phot_g_mean_mag'],
                   tess['Tmag'], gaia['pmra'], gaia['pmdec'], gaia['parallax']])

    t.remove_row(0)
    return t


def tic_by_contamination(pos, r, contam, tmag_lim):
    """
    Allows the user to perform a counts only query or get the usual grid of results. When unsure
    how many results is expcted, it is best to first perform a counts query to avoid memory overflow
    Parameters
    ----------
        pos: [RA,Dec] pair to be the center of the search
        r: radius of cone search
        contam: [min,max] list of how much allowed contamination
    Returns
    ----------
        json.loads(outString): a table of source(s) in radius
    """
    request = {'service':'Mast.Catalogs.Filtered.Tic.Position',
               'format':'json',
               'params': {'columns':'c.*',
                          'filters': [{'paramName':'contratio',
                                       'values':[{'min':contam[0], 'max':contam[1]}]},
                                      {'paramName':'Tmag',
                                       'values':[{'min':0, 'max':tmag_lim}]}
                                      ],
                          'ra':pos[0],
                          'dec':pos[1],
                          'radius':r
                          }}
    headers, outString = mastQuery(request)
    return jsonTable(json.loads(outString))
