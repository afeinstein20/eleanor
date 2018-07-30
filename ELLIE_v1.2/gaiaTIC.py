import sys, os, time, re, json
import numpy as np
import astropy.units as u
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
from astropy.table import Table
from astroquery.mast import Catalogs

try: # Python 3.x
    from urllib.parse import quote as urlencode
    from urllib.request import urlretrieve
    import http.client as httplib
except ImportError: # Python 2.x
    from urllib import pathname2url as urlencode
    from urllib import urlretrieve
    import httplib

# This is something used by both
def mastQuery(request):
    """
    Sends a request to the MAST server
    Parameters
    ----------
        request: json string
    Returns
    ----------
        head: Headers for response
        content: Data for response
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


# This is something used by the user
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


# This is something used by the whole FFI
def coneSearch(pos, r, service):
    """
    Completes a cone search in the Gaia DR2 or TIC
    Parameters
    ----------
        pos: [RA,Dec] list
        r: radius of cone search, [deg]
        service: identifies which MAST service to use. Either: 'Mast.Catalogs.GaiaDR2.Cone'
                 or 'Mast.Catalogs.Tic.Cone' are acceptable inputs
    Returns
    ----------
        json.loads(outString): dictionary of sources in cone search
        See the Gaia field documentation for information on returned columns
    """
    request = {'service': service,
               'params': {'ra':pos[0], 'dec':pos[1], 'radius':r},#/3600.},
               'format':'json'}
    headers, outString = mastQuery(request)
    return json.loads(outString)


# This is something used by a user
def crossmatch(pos, r, service):
    """
    Crossmatches [RA,Dec] position to a source in the Gaia DR2 catalog or TIC catalog
    Parameters
    ----------
        pos: [RA,Dec] list
        r: radius of search for crossmatch
        service: identifies which catalog to crossmatch to. Either: 'Mast.GaiaDR2.Crossmatch'
                 or 'Mast.Tic.Crossmatch' are accepted.
    Returns
    ----------
        json.loads(outString): dictionary of source(s) in radius of crossmatch
    """
    # This is a json object
    crossmatchInput = {'fields': [{'name':'ra' , 'type':'float'},
                                  {'name':'dec', 'type':'float'}],
                       'data': [{'ra':pos[0], 'dec':pos[1]}]}
    request = {'service':service,
               'data':crossmatchInput,
               'params': {'raColumn':'ra', 'decColumn':'dec', 'radius':r/3600.},
               'format':'json'}
    headers, outString = mastQuery(request)
    return json.loads(outString)['data'][0]



# This is something used by the whole FFI
def ticSearchByContam(pos, r, contam):
    """
    Allows the user to perform a counts only query or get the usual grid of results. When unsure
    how many results is expected, it is best to first perform a counts query to avoid memory overflow
    Parameters
    ---------- 
        pos: [RA,Dec] pair
        r: radius of cone search
        contam: [min,max] list of how much allowed contamination
    Returns
    ---------- 
        json.loads(outString): dictionary of source(s) in radius
    """
    request = {'service':'Mast.Catalogs.Filtered.Tic.Position',
               'format':'json',
               'params': {'columns':'c.*',
                          'filters': [{'paramName':'contratio', 
                                       'values':[{'min':contam[0], 'max':contam[1]}]}],
                          'ra':pos[0],
                          'dec':pos[1],
                          'radius':r
                          }}
    headers, outString = mastQuery(request)
    return json.loads(outString)


# This is something used by the user
def gaiaPositionByID(source_id):
    """
    Finds the RA,Dec for a given Gaia source_id
    Parameters
    ---------- 
        source_id: Gaia source_id (float)
    Returns
    ---------- 
        source_id, ra, dec, gmag, pmra, pmdec, parallax
    """
    adql = 'SELECT gaia.source_id, gaia.ra, gaia.dec, gaia.phot_g_mean_mag, gaia.pmra, gaia.pmdec, gaia.parallax FROM gaiadr2.gaia_source AS gaia WHERE gaia.source_id={0}'.format(source_id)
    job = Gaia.launch_job(adql)
    table = job.get_results()
    return table['source_id'].data, [table['ra'].data[0], table['dec'].data[0]], table['phot_g_mean_mag'].data, table['pmra'].data[0], table['pmdec'].data[0], table['parallax'].data[0]


# This is something used by the user
def ticPositionByID(tic_id):
    """
    Finds the RA,Dec for a given TIC source_id
    Parameters
    ----------
        tic_id: TIC source_id (float)
    Returns
    ----------
        source_id, ra, dec, tmag
    """
    ticData = Catalogs.query_criteria(catalog='Tic', ID=tic_id)
    print(ticData['ID'].data, [ticData['ra'].data[0], ticData['dec'].data[0]], ticData['Tmag'].data)
    return ticData['ID'].data, [ticData['ra'].data[0], ticData['dec'].data[0]], ticData['Tmag'].data



def makeTable():
    """ Creates a table for crossmatching multiple sources between Gaia and TIC catalogs """
    columns = ['Gaia_ID', 'TIC_ID', 'RA', 'Dec', 'separation', 'Gmag', 'Tmag', 'pmra', 'pmdec', 'parallax']
    t = Table(np.zeros(10), names=columns)
    t['RA'].unit, t['Dec'].unit, t['separation'].unit = u.arcsec, u.arcsec, u.arcsec
    t['pmra'].unit, t['pmdec'].unit = u.mas/u.year, u.mas/u.year
    return t


# This is something used by the user
def gaiaMultiCrossmatch(filename):
    """
    Allows the user to pass in a file of Gaia source_IDs to be crossmatched with TIC
    Parameters
    ---------- 
        filename: name of file containing IDs. Each ID should be a different row.
    Returns
    ---------- 
        table: Table of gaia_ID, TIC_ID, RA, Dec, delta_RA, delta_Dec, Gmag, Tmag, pmra, pmdec, parallax
    """
    sources = np.loadtxt(filename, dtype=int)
    service = 'Mast.Tic.Crossmatch'
    t = makeTable()
    for i in sources:
        sID, pos, gmag, pmra, pmdec, parallax = gaiaPositionByID(i)
        tic = crossmatch(pos, 0.01, service)
        pos[0], pos[1] = (pos[0]*u.deg), (pos[1]*u.deg)
        c1 = SkyCoord(pos[0], pos[1], frame='icrs')
        c2 = SkyCoord(tic['MatchRA']*u.deg, tic['MatchDEC']*u.deg, frame='icrs')
        print(c1.separation(c2))
        separation = c1.separation(c2).to(u.arcsec)
        t.add_row([i, tic['MatchID'], (pos[0]).to(u.arcsec), (pos[1]).to(u.arcsec), separation, gmag, tic['Tmag'], pmra, pmdec, parallax])
    t.remove_row(0)
    print(t)
    return t


# This is something used by the user
def ticMultiCrossmatch(filename):
    """
    Allows the user to pass in a file of TIC IDs to be crossmatched with Gaia
    Parameters
    ----------
        filename: name of file containing IDs. Each ID should be a different row.
    Returns
    ----------
        table: Table of gaia_ID, TIC_ID, RA, Dec, delta_RA, delta_Dec, Gmag, Tmag, pmra, pmdec, parallax
    """
    sources = np.loadtxt(filename, dtype=int)
    service = 'Mast.GaiaDR2.Crossmatch'
    t = makeTable()
    for i in sources:
        tID, pos, tmag = ticPositionByID(i)
        gaia = crossmatch(pos, 0.01, service)
        pos[0],pos[1] = (pos[0]*u.deg), (pos[1]*u.deg)
        c1 = SkyCoord(pos[0], pos[1], frame='icrs')
        c2 = SkyCoord(gaia['MatchRA']*u.deg, gaia['MatchDEC']*u.deg, frame='icrs')
        separation = c1.separation(c2).to(u.arcsec)
        t.add_row([gaia['MatchID'], i, (pos[0]).to(u.arcsec), (pos[1]).to(u.arcsec), separation, gaia['phot_g_mean_mag'], 
                   tmag, gaia['pmra'], gaia['pmdec'], gaia['parallax']])
    
    t.remove_row(0)
    print(t)
    return t

# This is something used by the user
def findByPosition(filename):
    """
    Allows the user to pass in a file of RA, Dec to be matched with Gaia & TIC IDs
    Parameters
    ----------  
        filename: name of file containing RA,Dec pairs. RAs should be column 1 & matching Dec should be column 2
                  File can be either plain text file or CSV
    Returns
    ----------  
        table: Table of gaia_ID, TIC_ID, RA, Dec, delta_RA, delta_Dec, Gmag, Tmag, pmra, pmdec, parallax
    """
    if filename[-3::] == 'csv':
        data = np.loadtxt(filename, delimiter = ',')
    else:
        data = np.loadtxt(filename)

    t = makeTable()    
    for i in range(len(data)):
        pos = data[i]
        gaia = coneSearch(pos, 0.001, 'Mast.Catalogs.GaiaDR2.Cone')['data'][0]
        tess = crossmatch(pos, 0.01, 'Mast.Tic.Crossmatch')
        pos[0] = (pos[0]*u.deg).to(u.arcsec)
        pos[1] = (pos[1]*u.deg).to(u.arcsec)
        t.add_row([gaia['source_id'], tess['MatchID'], pos[0], pos[1], pos[0]-tess['MatchRA'], pos[1]-tess['MatchDEC'], gaia['phot_g_mean_mag'],
                   tess['Tmag'], gaia['pmra'], gaia['pmdec'], gaia['parallax']])
    t.remove_row(0)
    return t



#crossmatch([48.00658181793736, 29.008217615212036], 0.01, 'Mast.GaiaDR2.Crossmatch')
#coneSearch([48.00658181793736, 29.008217615212036], 800., 'Mast.Catalogs.Tic.Cone')
#gaiaMultiCrossmatch('testGaia.txt')
#ticMultiCrossmatch('testTIC.txt')
#findByPosition('testPosition.csv')
