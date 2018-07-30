import sys, os, time, re, json
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
from astropy.table import Table
from astroquery.mast import Catalogs
from lightkurve import KeplerTargetPixelFile as ktpf
from astropy.wcs import WCS
import matplotlib.animation as animation
import matplotlib
from astropy.io import fits
import requests
from bs4 import BeautifulSoup

#matplotlib.use("Agg")

try: # Python 3.x
    from urllib.parse import quote as urlencode
    from urllib.request import urlretrieve
    import http.client as httplib
except ImportError: # Python 2.x
    from urllib import pathname2url as urlencode
    from urllib import urlretrieve
    import httplib


class locate_sources(object):
    """
    Finds sources in Gaia or TIC catalogs
    User can pass in either ID or (RA,Dec)
    """

    def __init__(self, targetID=None, position=None):
        self.targetID = targetID
        self.position = position



    def mastQuery(self, request):
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
        server = 'mast.stsci.edu'
        # Grab Python version
        version = '.'.join(map(str, sys.version_info[:3]))
        # Create Http header variables
        headers = {'Content-type': 'application/x-www-form-urlencoded',
                   'Accept': 'text/plain',
                   'User-agent': 'python-requests/'+version}
        # Encoding the request as a json string
        requestString = urlencode(json.dumps(request))
        # Opening the https connection
        conn = httplib.HTTPSConnection(server)
        # Making the query
        conn.request('POST', '/api/v0/invoke', 'request='+requestString, headers)
        # Getting the response
        resp    = conn.getresponse()
        head    = resp.getheaders()
        content = resp.read().decode('utf-8')
        # Closes the https connection
        conn.close()
        return head, content
    

    def jsonTable(jsonObj):
        """
        Converts json return type object into an astropy Table
        Parameters
        ---------- 
            jsonObj: an object from mastQuery
        Returns
        ---------- 
            table: astropy table
        """
        dataTable = Table()
        for col,atype in [(x['name'],x['type']) for x in jsonObj['fields']]:
            if atype=='string':
                atype='str'
            if atype=='boolean':
                atype='bool'
            if atype=='int':
                atype='float'
            dataTable[col] = np.array([x.get(col,None) for x in jsonObj['data']],dypte=atype)
        return dataTable


    def cone_search(pos, r, service):
        """
        Completes a cone search in the GaiaDR2 or TIC catalogs
        Parameters
        ----------
            pos: [RA,Dec] center point
            r: radius of cone search (deg)
            service: identifies which MAST service to use; either 'Mast.Catalogs.GaiaDR2.Cone' 
                     or 'Mast.Catalogs.Tic.Cone' are acceptable inputs
        Returns
        ----------  
            table: table of output values; see either Gaia DR2 or TIC field documentation for 
                   more information on returned columns
        """
        request = {'service':service,
                   'params': {'ra':pos[0], 'dec':pos[1], 'radius':r},
                   'format': 'json'}
        headers, outString = mastQuery(request)
        data = json.loads(outString)
        return jsonTable(data)
           
    
    
    def crossmatch(pos, r, service):
        """
        Crossmatches (RA,Dec) position to a source in the Gaia DR2 catalog or TIC catalog
        Parameters
        ---------- 
            pos: (RA, Dec) pair. Only accepts one pair at a time
            r: radius of search crossmatch; defaults to r=0.01
            service: identifies which catalog to crossmatch to. Either 'Mast.GaiaDR2.Crossmatch'
                     or 'Mast.Tic.Crossmatch' are accepted
        Returns
        ---------- 
            table: table of source(s) in radius of crossmatch
        """
        # This is a json object
        crossmatchInput = {'fields': [{'name':'ra' , 'type':'float'},
                                      {'name':'dec', 'type':'float'}],
                           'data': [{'ra':pos[0], 'dec':pos[1]}]}
        request = {'service':service,
                   'data':crossmatchInput,
                   'params': {'raColumn':'ra', 'decColumn':'dec', 'radius':r},
                   'format':'json'}
        headers, outString = mastQuery(request)
        data = json.loads(outString)['data']
        return jsonTable(data)

    
    def by_gaia_id(source_id):
        """
        Finds the (RA,Dec) for a given Gaia source_id
        Parameters 
        ---------- 
            source_id: Gaia source_id (float)
        Returns
        ---------- 
            table: table of information about source_id; check Gaia field
                   documentation for more information
        """
        adql = 'SELECT gaia.source_id, gaia.ra, gaia.dec, gaia.phot_g_mean_mag, gaia.pmra, gaia.pmdec, gaia.parallax FROM gaiadr2.gaia_source AS gaia WHERE gaia.source_id={0}'.format(source_id)
        job = Gaia.launch_job(adql)
        table = job.get_results()
        return table['source_id'].data, [table['ra'].data[0], table['dec'].data[0]], table['phot_g_mean_mag'].data, table['pmra'].data[0], table['pmdec'].data[0], table['parallax'].data[0]


    def by_tic_id(tic_id):
        """
        Finds the (RA,Dec) for a given TIC source_id
        Parameters
        ---------- 
            tic_id: TIC source_id (float)
        Returns
        ----------
            table: table of information about tic_id; check TIC field
                   documentation for more information
        """
        ticData = Catalogs.query_criteria(catalog='Tic', ID=tic_id)
        return ticData['ID'].data, [ticData['ra'].data[0], ticData['dec'].data[0]], ticData['Tmag'].data

    
    def make_table():
        """ Creates a table for crossmatching multiple sources between Gaia & TIC catalogs """
        columns = ['Gaia_ID', 'TIC_ID', 'RA', 'Dec', 'separation', 'Gmag', 'Tmag', 'pmra', 'pmdec', 'parallax']
        t = Table(np.zeros(10), names=columns)
        t['RA'].unit, t['Dec'].unit, t['separation'].unit = u.arcsec, u.arcsec, u.arcsec
        t['pmra'].unit, t['pmdec'].unit = u.mas/u.year, u.mas/u.year
        return t


    def find_dist_diff(pos, match_ra, match_dec):
        """
        Finds separation distance when crossmatching objects or 
        looking for objects with a given (RA,Dec)
        Returns 
        ---------- 
            separation: separation distance in arcsec
        """
        pos[0], pos[1] = (pos[0]*u.deg), (pos[1]*u.deg)
        c1 = SkyCoord(pos[0], pos[1], frame='icrs')
        c2 = SkyCoord(match_ra*u.deg, match_dec*u.deg, frame='icrs')
        return c1.separation(c2).to(u.arcsec)


    def multi_gaia_to_tic(filename):
        """
        Allows the user to pass in a file of Gaia source_IDs to be crossmatched with TIC
        Parameters
        ---------- 
            filename: name of file containing Gaia source_IDs; each ID should be a different row
        Returns
        ---------- 
            table: table of gaia_ID, TIC_ID, RA, Dec, delta_pos, Gmag, Tmag, pmra, pmdec, parallax
        """
        sources = np.loadtxt(filename, dtype=int)
        service = 'Mast.Tic.Crossmatch'
        t = makeTable()
        for s in sources:
            sID, pos, gmag, pmra, pmdec, parallax = by_gaia_id(s)
            tic = crossmatch(pos, 0.01, service)
            separation = find_dist_diff(pos, tic['MatchRA'], tic['MatchDEC'])
            t.add_row([i, tic['MatchID'], (pos[0]).to(u.arcsec), (pos[1]).to(u.arcsec), separation, gmag, tic['Tmag'], pmra, pmdec, parallax])
        t.remove_row(0)
        return t


    def multi_tic_to_gaia(filename):
        """
        Allows the user to pass in a file of TIC IDs to be crossmatched with Gaia
        Parameters
        ----------  
            filename: name of file containing TIC IDs; each ID should be a different row
        Returns
        ----------  
            table: table of gaia_ID, TIC_ID, RA, Dec, delta_pos, Gmag, Tmag, pmra, pmdec, parallax
        """
        sources = np.loadtxt(filename, dtype=int)
        service = 'Mast.GaiaDR2.Crossmatch'
        t = makeTable()
        for s in sources:
            tID, pos, tmag = by_tic_id(s)
            gaia = crossmatch(pos, 0.01, service)
            separation =  find_dist_diff(pos, gaia['MatchRA'], gaia['MatchDEC'])
            t.add_row([gaia['MatchID'], i, (pos[0]).to(u.arcsec), (pos[1]).to(u.arcsec), separation, gaia['phot_g_mean_mag'],
                       tmag, gaia['pmra'], gaia['pmdec'], gaia['parallax']])
        t.remove_row(0)
        return t


    def find_by_raDec(filename):
        """
        Allows the user to pass in a file of RA,Dec to be matched with Gaia & TIC IDs
        Parameters
        ---------- 
            filename: name of file containing RA,Dec pairs; RAs should be column 1 * matching Dec
                      should be column 2; file can be either plain text or .csv
        Returns
        ---------- 
            table: table of gaia_ID, TIC_ID, RA, Dec, delta_pos, Gmag, Tmag, pmra, pmdec, parallax
        """
        if filename[-3::]=='csv':
            data = np.loadtxt(filename, delimiter=',')
        else:
            data = np.loadtxt(filename)
        
        t = makeTable()
        for i in range(len(data)):
            pos = data[i]
            gaia = cone_search(pos, 0.001, 'Mast.Catalogs.GaiaDR2.Cone')['data'][0]
            tess = crossmatch(pos, 0.01 , 'Mast.Tic.Crossmatch')
            separation = find_dist_diff(pos, tess['MatchRA'], tess['MatchDEC'])
            t.add_row([gaia['source_id'], tess['MatchID'], pos[0], pos[1], pos[0]-tess['MatchRA'], pos[1]-tess['MatchDEC'], gaia['phot_g_mean_mag'],
                   tess['Tmag'], gaia['pmra'], gaia['pmdec'], gaia['parallax']])
        t.remove_row(0)
        return t


    def find_files_in_dir(dir):
        """ Finds FITS files in a given directory """
        files = np.array(os.listdir(dir))
        fitsInds = np.array([i for i,item in enumerate(files) if 'fits' in item])
        return filenames[fitsInds]


    def find_camera_chip(pos, dir):
        """ 
        Uses the RA,Dec of each camera's chips and the position of the source to find
        which files the source is located in
        Parameters
        ---------- 
            pos: (RA,Dec) pair of the location of the source
        Returns
        ---------- 
            camera, chip: the camera # and chip # pair where the source is located
        """
        for camera in np.arange(1,5,1):
            for chip in np.arange(1,5,1):
                dir = './calFITS_2019-{}-{}/'.format(camera, chip)
                fns = find_files(dir)
                mast, mheader = fits.getdata(dir+fns[0], header=True)
                xy = WCS(mheader).all_world2pix(pos[0], pos[1], 1, quiet=True)
                if (xy[0] >= 0.) and (xy[0] <= len(mast)) and (xy[1] >= 0.) and (xy[1] <= len(mast[0])):
                    return camera, chip
        return "No camera and chip pair found."


    def find_all_fits(year, camera, chip, dayMin, dayMax):
        """
        Finds all calibrated fits files from URL
        Parameters 
        ---------- 
            year, camera, chip, dayMin, dayMax: basic parameters for determining 
                  URL to download FITS files from
        Returns
        ---------- 
            calFiles: list of calibrated filenames from URL
        """
        calFiles = []
        days = np.arange(dayMin, dayMax+1, 1)
        url = 'https://archive.stsci.edu/missions/tess/ete-6/ffi/'
        for d in days:
            path = url + '/' + year + '/' + str(d) + '/' + camera + '-' + chip + '/'
            for fn in BeautifulSoup(requests.get(path).text, 'lxml').find_all('a'):
                if fn.get('href')[-7::] == 'ic.fits':
                    calFiles.append(path + fn.get('href'))
        return calFiles, url


    def download_files(year, camera, chip, dayMin, dayMax):
        """ Downloads all FITS files for a given year, camera, chip, and days
        Parameters
        ---------- 
            year, camera, chip, dayMin, dayMax: basic parameters for determining
                  URL to download FITS files from
        Returns
        ---------- 
        """
        dir = './calFITS_2019_{}-{}/'.format(camera, chip)
        fns, url = find_all_fits(year, camera, chip, dayMin, dayMax)
        if os.path.isdir(dir) == False:
            os.mkdir(dir)
        for i in fns:
            os.system('cd {} && curl -O -L {}'.format(dir, i))
        return

    
    def search_tic_by_contam(self, pos, r, contam):
        """
        Allows the user to perform a counts only query or get the usual grid of results. When unsure
        how many results is expected, it is best to first perform a counts query to avoid memory overflow
        Parameters 
        ----------
            pos: (RA,Dec) pair
            r: radius of cone search
            contam: [min, max] list of how much contamination is allowed
        Returns
        ----------
            table: table of sources in cone radius
        """
        request = {'service': 'Mast.Catalogs.Filtered.Tic.Position',
                   'format': 'json',
                   'params': {'columns': 'c.*',
                              'filters': [{'paramName': 'contratio',
                                           'values':[{'min':contam[0], 'max':contam[1]}]}],
                              'ra':pos[0],
                              'dec':pos[1],
                              'radius':r
                              }}
        headers, outString = self.mastQuery(request)
        data = json.loads(outString)
        return jstonTable(data)


    def radec_to_pixel(header, pos):
        """ Converts (RA,Dec) -> (x,y) given FITS file """
        return WCS(header).all_world2pix(ra=pos[0], dec=pos[1], 1)
    

    def sort_by_date(dir):
        """
        Sorts FITS files by start date of observation
        Parameters
        ----------
            dir: directory the FITS files are in
        Returns
        ----------
            fns: sorted filenames by date
        """
        dates = []
        fns = self.find_files_in_dir(dir)
        for f in fns:
            mast, header = fits.getdata(dir+f, header=True)
            dates.append(header['DATE-OBS'])
        dates, fns = np.sort(np.array([dates, fns]))
        return fns

    
    def create_stacked_postcards(dir):
        """
        Creates n x m postcards (i.e. breakdowns of chip FFI) and 
        saves to a new directory
        Parameters
        ----------
            dir: directory in which the camera-chip FFIs are located
        Returns
        ----------
        """
        return



    def create_pointing_model(dir):
        """
        Creates a pointing model for a given postcard
        Parameters
        ----------
            dir: directory in which the FITS files can be found
        Returns
        ----------
        """
        fns = self.sort_by_date(dir)
        
        mast, header = fits.getdata(fns[0], header=True)
        chip_center  = [header['CRVAL1'], header['CRVAL2']]
        objs = self.search_tic_by_contam(self, chip_center, 6*np.sqrt(2), [0.0, 0.01])
        

        def calc_shift():
            global dir, fns, x, y, corrFile
            """
            Calculates the deltaX, deltaY, and rotation for isolated sources in order to
            put together a pointing model for the entire chip
            Parameters
            ----------
            Returns
            ----------
            """
            return
