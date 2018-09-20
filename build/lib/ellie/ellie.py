"""
ELLIE
-----

This is ELLIE.

"""

import os, sys, re, json, time
import requests
import urllib
from bs4 import BeautifulSoup
import numpy as np
from time import strftime
from tqdm import tqdm
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astroquery.mast import Catalogs
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table, Column, Row
from astropy.wcs import WCS
from astropy.nddata import Cutout2D

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.widgets import RadioButtons

import mplcursors

from photutils import CircularAperture, RectangularAperture, aperture_photometry
from muchbettermoments import quadratic_2d
from lightkurve import KeplerTargetPixelFile as ktpf
from lightkurve import SFFCorrector
from scipy import ndimage
from scipy.optimize import minimize
import collections

from bokeh.palettes import Viridis256
from bokeh.io import push_notebook, show, output_notebook
from bokeh.models import (ColumnDataSource, HoverTool, BasicTicker,
                          Slider, Button, Label, LinearColorMapper, Span,
                          ColorBar)
from bokeh.plotting import figure, show, output_file
from bokeh.models   import ColumnDataSource, LinearColorMapper, ColorBar, BasicTicker

try: # Python 3.x
    from urllib.parse import quote as urlencode
    from urllib.request import urlretrieve
    import http.client as httplib
except ImportError: # Python 2.x
    from urllib import pathname2url as urlencode
    from urllib import urlretrieve
    import httplib

##########################
##########################
##########################
class find_sources:
    """
    The main interface to the ELLIE image extraction package

    Args:
        tic : The TESS Input Catalog identifier for the target in question (default: None)
        gaia: The Gaia identifier for the target in question (default: None)
        pos : The (RA, Dec) coordinates for the location in question (default: None)
        multiFile: Filename with either a list of TIC IDs, Gaia IDs, or RA,Dec position pairs (default:None)
    """

    def __init__(self, tic=None, gaia=None, pos=None, multiFile=None, dir=None, camera=None, chip=None):
        """ USER INPUTS """
        if tic == None:
            self.tic = 0
        else:
            self.tic = tic
        if gaia == None:
            self.gaia = 0
        else:
            self.gaia = gaia
        if pos == None:
            self.pos = []
        else:
            self.pos = pos
        if multiFile == None:
            self.mutliFile = []
        else:
            self.multiFile = multiFile
        if dir == None:
            self.dir = ''
        else:
            self.dir = dir
        if camera==None:
            self.camera=0
        else:
            self.camera=camera
        if chip==None:
            self.chip=0
        else:
            self.chip=chip



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


    def jsonTable(self, jsonObj):
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


    def cone_search(self, r, service, multiPos=None):
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
        elif self.pos == 0:
            return "No position found. Please try reinitializing."
        else:
            pos = self.pos
        request = {'service': service,
                   'params': {'ra':pos[0], 'dec':pos[1], 'radius':r},
                   'format':'json'}
        headers, outString = self.mastQuery(request)
        return self.jsonTable(json.loads(outString))


    def crossmatch_by_position(self, r, service, multiPos=None):
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
        # This is a json object
        if multiPos != None:
            pos = multiPos
        else:
            pos = self.pos
        crossmatchInput = {'fields': [{'name':'ra' , 'type':'float'},
                                      {'name':'dec', 'type':'float'}],
                           'data': [{'ra':pos[0], 'dec':pos[1]}]}
        request = {'service':service,
                   'data':crossmatchInput,
                   'params': {'raColumn':'ra', 'decColumn':'dec', 'radius':r},
                   'format':'json'}
        headers, outString = self.mastQuery(request)
        return self.jsonTable(json.loads(outString))



    def gaia_pos_by_ID(self, multiSource=None):
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
        elif self.gaia == 0:
            return "No Gaia source_id entered. Please try initializing again."
        else:
            source = self.gaia

        adql = 'SELECT gaia.source_id, gaia.ra, gaia.dec, gaia.phot_g_mean_mag, gaia.pmra, gaia.pmdec, gaia.parallax FROM gaiadr2.gaia_source AS gaia WHERE gaia.source_id={0}'.format(source)

        job = Gaia.launch_job(adql)
        table = job.get_results()
        return table


    def tic_pos_by_ID(self, multiSource=None):
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
        elif self.tic == 0:
            return "No TIC source_id entered. Please try again."
        else:
            source = self.tic

        ticData = Catalogs.query_criteria(catalog='Tic', ID=source)
        return ticData['ID'].data, [ticData['ra'].data[0], ticData['dec'].data[0]], ticData['Tmag'].data


    def initialize_table(self):
        """ Creates a table for crossmatching multiple sources between Gaia and TIC catalogs """
        columns = ['Gaia_ID', 'TIC_ID', 'RA', 'Dec', 'separation', 'Gmag', 'Tmag', 'pmra', 'pmdec', 'parallax']
        t = Table(np.zeros(10), names=columns)
        t['RA'].unit, t['Dec'].unit, t['separation'].unit = u.arcsec, u.arcsec, u.arcsec
        t['pmra'].unit, t['pmdec'].unit = u.mas/u.year, u.mas/u.year
        return t

    def crossmatch_distance(self, pos, match):
        """ Finds distance between source and crossmatched source(s) """
        c1 = SkyCoord(pos[0], pos[1], frame='icrs')
        c2 = SkyCoord(match[0]*u.deg, match[1]*u.deg, frame='icrs')
        return c1.separation(c2).to(u.arcsec)


    def crossmatch_multi_to_gaia(self):
        """
        Crossmatches file of TIC IDs to Gaia
        Parameters
        ----------
        Returns
        ----------
            table: table of gaia_id, tic_id, ra, dec, delta_pos, gmag, tmag, pmra, pmdec, parallax
        """
        filename = self.multiFile
        sources  = np.loadtxt(filename, dtype=int)
        service  = 'Mast.GaiaDR2.Crossmatch'
        r = 0.01
        t = self.initialize_table()
        for s in sources:
            tID, pos, tmag = self.tic_pos_by_ID(s)
            gaia = self.crossmatch_by_position(r, service, pos)
            pos[0], pos[1] = pos[0]*u.deg, pos[1]*u.deg
            separation = self.crossmatch_distance(pos, [gaia['MatchRA'], gaia['MatchDEC']])
            t.add_row([gaia['MatchID'], s, (pos[0]).to(u.arcsec), (pos[1]).to(u.arcsec), separation, gaia['phot_g_mean_mag'], tmag, gaia['pmra'], gaia['pmdec'], gaia['parallax']])
        t.remove_row(0)
        return t


    def crossmatch_multi_to_tic(self, list=[]):
        """
        Crossmatches file of Gaia IDs to TIC
        Parameters
        ----------
        Returns
        ----------
            table: table of gaia_id, tic_id, ra, dec, delta_pos, gmag, tmag, pmra, pmdec, parallax
        """
        if len(list) == 0:
            try:
                sources  = np.loadtxt(self.multiFile, dtype=int)
            except AttributeError:
                print("If calling from data_products: No filename found. Please make sure you spelled it correctly.")
                print("If calling from visualize.mark_gaia: No Gaia sources were found in this TPF.")
                return
        else:
            sources = list
        service  = 'Mast.Tic.Crossmatch'
        r = 0.1
        t = self.initialize_table()
        for s in sources:
            self.gaia = s
            table = self.gaia_pos_by_ID()
            sID, pos, gmag = table['source_id'].data, [table['ra'].data[0], table['dec'].data[0]], table['phot_g_mean_mag'].data
            pmra, pmdec, parallax = table['pmra'].data, table['pmdec'].data, table['parallax'].data
            self.pos = pos
            tic = self.crossmatch_by_position(r, service)
            self.pos = [self.pos[0]*u.deg, self.pos[1]*u.deg]
            separation = self.crossmatch_distance(self.pos, [tic['MatchRA'], tic['MatchDEC']])
            min = separation.argmin()
            row = tic[min]
            t.add_row([sID, row['MatchID'], (self.pos[0]).to(u.arcsec), (self.pos[1]).to(u.arcsec), separation[min], gmag, row['Tmag'], pmra, pmdec, parallax])
        t.remove_row(0)

        return t


    def find_by_position(self):
        """
        Allows the user to pass in a file of RA,Dec pairs to be matched in Gaia & TIC
        Parameters
        ----------
        Returns
        ----------
            table: table of gaia_id, tic_id, ra, dec, delta_pos_gaia, delta_pos_tic, gmag, tmag, pmra, pmdec, parallax
        """
        if self.pos == None:
            if self.multiFile[-3::] == 'csv':
                data = np.loadtxt(self.multiFile, delimiter=',')
            else:
                data = np.loadtxt(self.multiFile)

        else:
            data = [self.pos]

        columns = ['Gaia_ID', 'TIC_ID', 'RA', 'Dec', 'Gaia_sep', 'TIC_sep', 'Gmag', 'Tmag', 'pmra', 'pmdec', 'parallax']
        t = Table(np.zeros(11), names=columns)
        t['RA'].unit, t['Dec'].unit, t['Gaia_sep'].unit, t['TIC_sep'].unit = u.arcsec, u.arcsec, u.arcsec, u.arcsec
        t['pmra'].unit, t['pmdec'].unit = u.mas/u.year, u.mas/u.year

        for i in range(len(data)):
            pos = data[i]
            gaia = self.crossmatch_by_position(0.01, 'Mast.GaiaDR2.Crossmatch', pos)[0]
            tess = self.crossmatch_by_position(0.5, 'Mast.Tic.Crossmatch', pos)[0]
            pos[0], pos[1] = pos[0]*u.deg, pos[1]*u.deg
            gaiaPos = [gaia['MatchRA'], gaia['MatchDEC']]
            sepGaia = self.crossmatch_distance(pos, gaiaPos)
            tessPos = [tess['MatchRA'], tess['MatchDEC']]
            sepTess = self.crossmatch_distance(pos, tessPos)

            t.add_row([gaia['MatchID'], tess['MatchID'], pos[0], pos[1], sepGaia, sepTess, gaia['phot_g_mean_mag'],
                       tess['Tmag'], gaia['pmra'], gaia['pmdec'], gaia['parallax']])

        t.remove_row(0)
        return t

    def tic_by_contamination(self, pos, r, contam):
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
                                       'values':[{'min':contam[0], 'max':contam[1]}]}],
                          'ra':pos[0],
                          'dec':pos[1],
                          'radius':r
                          }}
        headers, outString = self.mastQuery(request)
        return self.jsonTable(json.loads(outString))


    def download_tic_tpf(self, custom=False):
        """
        This function finds the sector, camera, and chip a target is located in
        Downloads the already created stacked cadence FITS file (TPF) and associated
            light curve FITS file
        Downloads the postcard a target is located in if the target is not in the TIC
        Parameters
        ----------
            custom: Allows the user to ask to create TPF & light curve FITS files
                    for sources not in the TIC
        Returns
        ----------
            Downloads TPF & light curve FITS files
        """
        return




##########################
##########################
##########################
class data_products(find_sources):
    def __init__(self, tic=None, gaia=None, pos=None, dir=None):
        self.tic  = tic
        self.pos  = pos
        self.gaia = gaia

        if self.tic != None:
            self.tic_fn = 'hlsp_ellie_tess_ffi_{}_v1_lc.fits'.format(self.tic)
            id, pos, tmag = data_products.tic_pos_by_ID(self)
            self.pos = pos
        if self.gaia != None:
            self.gaia_fn = 'hlsp_ellie_tess_ffi_GAIA{}_v1_lc.fits'.format(self.gaia)

        """ Sets default directory to a local hidden directory .ellie"""
        """      with potential subdirectories ffis & postcards      """
        self.root_dir = './.ellie'
        if os.path.isdir(self.root_dir) == False:
            os.mkdir(self.root_dir)


    def download_ffis(self, sector=None, camera=None, chips=None):
        """ Downloads entire sector of data into .ellie/ffis/sector directory """
        def findAllFFIs(ca, ch):
            nonlocal year, days, url
            calFiles = []
            for d in days:
                path = '/'.join(str(e) for e in [url, year, d, ca])+'-'+str(ch)+'/'
                for fn in BeautifulSoup(requests.get(path).text, "lxml").find_all('a'):
                    if fn.get('href')[-7::] == 'ic.fits':
                        calFiles.append(path+fn.get('href'))
            return calFiles

        for s in sector:
            ffi_dir, sect_dir = 'ffis', 'sector_{}'.format(sector)
            # Creates an FFI directory, if one does not already exist
            if os.path.isdir(self.root_dir+'/'+sect_dir) == False:
                os.system('cd {} && mkdir {}'.format(self.root_dir, sect_dir))
            # Creates a sector directory, if one does not already exist
            if os.path.isdir(self.root_dir+'/'+sect_dir+'/'+ffi_dir) == False:
                os.system('cd {} && mkdir {}'.format(self.root_dir+'/'+sect_dir, ffi_dir))

            self.sect_dir = '/'.join(str(e) for e in [self.root_dir, sect_dir, ffi_dir])

            if sector in np.arange(1,14,1):
                year=2019
            else:
                year=2020
            # Current days available for ETE-6
            days = np.arange(129,132,1)
            if camera==None:
                camera = np.arange(1,5,1)
            if chips==None:
                chips  = np.arange(1,5,1)
            # This URL applies to ETE-6 simulated data ONLY
            url = 'https://archive.stsci.edu/missions/tess/ete-6/ffi/'
            for c in camera:
                for h in chips:
                    files = findAllFFIs(c, h)
                    # Loops through all files from MAST
                    for f in files:
                        file = Path(self.sect_dir+f)
                        # If the file in that directory doesn't exist, download it
                        if file.is_file() == False:
                            os.system('cd {} && curl -O -L {}'.format(self.sect_dir, f))
        return


    def sort_by_date(self, camera, chip):
        """ Sorts FITS files by start date of observation """
        fns = np.array(os.listdir(self.ffi_dir))
        pair = '{}-{}'.format(camera, chip)
        ffisInd = np.array([i for i,item in enumerate(fns) if 'ffic' and pair in item])
        camchip = np.array([i for i,item in enumerate(fns) if pair in item])
        fitsInd = [i for i in ffisInd if i in camchip]

        fns = fns[fitsInd]
        dates, time = [], []
        for f in fns:
            mast, header = fits.getdata(self.ffi_dir+f, header=True)
            dates.append(header['DATE-OBS'])
            time.append((header['TSTOP']-header['TSTART'])/2.)
        dates, fns = np.sort(np.array([dates, fns]))
        dates, time = np.sort(np.array([dates, time]))
        return fns, time


    def pointing_model(self, camera=None, chip=None, sector=None):
        """
        Creates the pointing model for a given camera and chip across all cadences
        """

        def pixel_cone(header, r, contam):
            """ Completes a cone search around center of FITS file """
            pos = [header['CRVAL1'], header['CRVAL2']]
            data = find_sources.tic_by_contamination(find_sources(), pos, r, contam)
            ra, dec = data['ra'], data['dec']
            return WCS(header).all_world2pix(ra, dec, 1), data['ID'], data['Tmag']


        def find_isolated(x, y):
            """
            Finds isolated sources in the image where an isolated source is >= 8.0 pixels
            away from any other source
            Parameters
            ----------
                x: a list of x-coordinates for all sources in file
                y: a list of y-coordinates for all sources in file
            Returns
            ----------
                isolated: a list of isolated source indices
            """

            def nearest(x_source, y_source):
                """ Calculates distance to the nearest source """
                nonlocal x, y
                x_list = np.delete(x, np.where(x==x_source))
                y_list = np.delete(y, np.where(y==y_source))
                closest = np.sqrt( (x_source-x_list)**2 + (y_source-y_list)**2 ).argmin()
                return np.sqrt( (x_source-x_list[closest])**2+ (y_source-y_list[closest])**2 )

            isolated = []
            for i in range(len(x)):
                dist = nearest(x[i], y[i])
                if dist >= 8.0:
                    isolated.append(i)
            return isolated


        def calc_shift():
            nonlocal fns, x, y, mast
            """
            Calculates the deltaX, deltaY, and rotation shift for isolated sources in order to
            create a pointing model for the entire chip
            """
            def model(params):
                nonlocal xy, centroids
                theta, xT, yT = params

                xRot = xy[:,0]*np.cos(theta) - xy[:,1]*np.sin(theta) - xT
                yRot = xy[:,0]*np.sin(theta) + xy[:,1]*np.cos(theta) - yT

                dist = np.sqrt((xRot-centroids[:,0])**2 + (yRot-centroids[:,1])**2)
                return np.nansum(np.square(dist))

            fns = np.array([self.ffi_dir+'/'+i for i in fns])

            x_cen, y_cen = len(mast)/2., len(mast[0])/2.
            xy = np.zeros((len(x),2))
            matrix = np.zeros((len(x), len(fns), 2))

            for i in range(len(x)):
                xy[i][0] = x[i]-x_cen
                xy[i][1] = y[i]-y_cen

            with open(self.corrFile, 'w') as tf:
                tf.write('cadence medT medX medY\n')

            for j in range(len(fns)):
                image, header = fits.getdata(fns[j], header=True)
                for i in range(len(x)):
                    tpf = Cutout2D(image, position=(x[i], y[i]), size=(5,5), mode='partial')
                    com = ndimage.measurements.center_of_mass(tpf.data.T-np.median(tpf.data)) # subtracts bkg
                    if (np.isfinite(com[0]) == True) and (np.isfinite(com[1]) == True):
                        matrix[i][j][0] = com[0]+xy[i][0]
                        matrix[i][j][1] = com[1]+xy[i][1]

            for i in tqdm(range(len(fns))):
                centroids = matrix[:,i]
                if i == 0:
                    initGuess = [0.01, 1.0, -1.0]
                else:
                    initGuess = oldSol

                bnds = ((-0.08, 0.08), (-3.0, 3.0), (-3.0, 3.0))
                solution = minimize(model, initGuess, method='L-BFGS-B', bounds=bnds, options={'ftol':5e-11,
                                                                                               'gtol':5e-05})#, 'eps':1e-02})
                sol = solution.x
                with open(self.corrFile, 'a') as tf:
                    theta, delX, delY = sol[0], sol[1], sol[2]
                    tf.write('{}\n'.format(str(i) + ' ' + str(theta) + ' ' + str(delX) + ' ' + str(delY)))
                    oldSol = sol
            return

        self.ffi_dir = self.root_dir+'/sector_{}/ffis/'.format(sector)
        self.corrFile = 'pointingModel_{}_{}-{}.txt'.format(sector, camera, chip)

        fns, time = self.sort_by_date(camera, chip)
        mast, header = fits.getdata(self.ffi_dir+fns[0], header=True)
        print("Grabbing good sources for the pointing model")
        xy, ids, tmag = pixel_cone(header, 6*np.sqrt(1.2), [0.0, 5e-6])

        # Makes sure sources are in the file
        bord = 20.
        inds = np.where( (xy[1]>=44.+bord) & (xy[1]<len(mast)-45.-bord) & (xy[0]>=bord) & (xy[0]<len(mast[0])-41.-bord))[0]
        x, y, ids, tmag = xy[0][inds], xy[1][inds], ids[inds], tmag[inds]

        isolated = find_isolated(x, y)
        print("{} isolated sources will be used to create the pointing model".format(len(isolated)))
        x, y, ids, tmag = x[isolated], y[isolated], ids[isolated], tmag[isolated]

        # Applying Tmag cuts
        tmag_inds = np.where(tmag <= 14.0)[0]
        x, y, ids, tmag = x[tmag_inds], y[tmag_inds], ids[tmag_inds], tmag[tmag_inds]
        calc_shift()
        return


    def make_postcard(self, camera=None, chip=None, sector=None):
        """
        Creates 300 x 300 x n postcards, where n is the number of cadences for a given observing run
        Creates a catalog of the associated header with each postcard for use later
        """
        if camera==None or chip==None or sector==None:
            print("You must input camera, chip, and sector you wish to create postcards for.")
            print("You can do this by calling data_products.make_postcard(sector=#, camera=#, chip=#)")
            return

        self.post_dir = self.root_dir+'/sector_{}/postcards/'.format(sector)
        self.ffi_dir  = self.root_dir+'/sector_{}/ffis/'.format(sector)
        if os.path.isdir(self.post_dir)==False:
            os.system('cd {} && mkdir {}'.format(self.root_dir+'/sector_{}/'.format(sector), 'postcards'))

        fns, time = self.sort_by_date(camera, chip)
        fns = [self.ffi_dir+i for i in fns]
        mast, mheader = fits.getdata(fns[0], header=True)
        x, y = np.linspace(45, 2093, 8, dtype=int), np.linspace(0, 2048, 8, dtype=int)
        x_cens, y_cens = [], []
        cat = 'postcard_catalog.txt'
        # Creating a table for the headers of each file
        colnames = list(mheader.keys())
        add  = ['POST_FILE', 'POST_SIZE1', 'POST_SIZE2', 'POST_CEN_X', 'POST_CEN_Y', 'POST_CEN_RA', 'POST_CEN_DEC']
        for a in add:
            colnames.append(a)

        typeList = np.full(len(colnames), 'S30', dtype='S30')
        t = Table(names=colnames, dtype=typeList)

        for i in range(len(x)-1):
            for j in range(len(y)-1):
                mast, mheader = fits.getdata(fns[0], header=True)
                fn = 'postcard_{}_{}-{}_{}-{}.fits'.format(sector, camera, chip, i, j)
                if os.path.isfile(self.post_dir+fn)==True:
                    return
                print("Creating postcard: {}".format(fn))
                x_cen = (x[i]+x[i+1]) / 2.
                y_cen = (y[j]+y[j+1]) / 2.

                radec = WCS(mheader).all_pix2world(x_cen, y_cen, 1)
                s = 300
                tpf   = ktpf.from_fits_images(images=fns, position=(x_cen,y_cen), size=(s,s))
                # Edits header of FITS files
                tempVals = list(mheader.values())
                moreData = [fn, s, s, x_cen, y_cen, float(radec[0]), float(radec[1])]
                for m in moreData:
                    tempVals.append(m)
                t.add_row(vals=tempVals)

                time_arrays = np.zeros((3, len(fns)))
                for f in range(len(fns)):
                    hdu = fits.open(fns[f])
                    hdr = hdu[1].header
                    time_arrays[0][f] = hdr['TSTART']
                    time_arrays[1][f] = hdr['TSTOP']
                    time_arrays[2][f] = hdr['BARYCORR']

                hdr = mheader
                hdr.append(('COMMENT', '***********************'))
                hdr.append(('COMMENT', '*     ELLIE INFO      *'))
                hdr.append(('COMMENT', '***********************'))
                hdr.append(('AUTHOR' , 'Adina D. Feinstein'))
                hdr.append(('VERSION', '1.0'))
                hdr.append(('GITHUB' , 'https://github.com/afeinstein20/ELLIE'))
                hdr.append(('CREATED', strftime('%Y-%m-%d'),
                            'ELLIE file creation date (YYY-MM-DD)'))
                hdr.append(('CEN_X'  , np.round(x_cen, 8)))
                hdr.append(('CEN_Y'  , np.round(y_cen, 8)))
                hdr.append(('CEN_RA' , float(radec[0])))
                hdr.append(('CEN_DEC', float(radec[1])))

                dtype = [
                    ("TSTART", np.float64),
                    ("TSTOP", np.float64),
                    ("BARYCORR", np.float64),
                    ("FLUX", np.float32, tpf.flux.shape[1:]),
                ]
                data = np.empty(len(tpf.flux), dtype=dtype)
                data["TSTART"] = time_arrays[0]
                data["TSTOP"] = time_arrays[1]
                data["BARYCORR"] = time_arrays[2]
                data["FLUX"] = tpf.flux

                hdu1 = fits.BinTableHDU.from_columns(fits.ColDefs(data), header=hdr)
                # hdu1 = fits.PrimaryHDU(header=hdr, data = new_postcard)
                hdu1.writeto(self.post_dir+fn)

        ascii.write(t, output='postcard_{}_{}-{}.txt'.format(sector, camera, chip))
        self.make_postcard_catalog()
        return


    def make_postcard_catalog(self):
        """
        Whenever a postcard is created for a camera-chip pair, a new catalog is created
            called: "postcard_{}-{}.txt".format(camera, chip)
        This function will take all of the postcard sub-catalogs and create a main one
            for each sector
        This file will be stored online and will be called from such, preventing the user
            from having to download it onto their personal machine
        Returns
        ----------
            postcard.txt: a catalog of header information for each postcard
        """
        from astropy.table import join

        output_fn = 'postcard.txt'
        dir_fns = np.array(os.listdir())
        post_fns = np.array([i for i,item in enumerate(dir_fns) if 'postcard_' in item])
        post_fns = dir_fns[post_fns]
        main_table = Table.read(post_fns[0], format='ascii.basic')
        for i in range(1,len(post_fns)):
            post = Table.read(post_fns[i], format='ascii.basic')
            for j in range(len(post)):
                main_table.add_row(post[j])

        ascii.write(main_table, output='postcard.txt')
        for i in post_fns:
            os.remove(i)
        return


    def find_postcard(self):
        """
        Finds what postcard a source is located in
        Returns
        ----------
            postcard filename, header of the postcard
        """
        t = self.get_header()

        if self.pos == None and self.tic != None:
            id, pos, tmag = data_products.tic_pos_by_ID(self)
            self.pos = pos

        in_file=[None]
        # Searches through rows of the table
        for i in range(len(t)):
            data=[]
            # Creates a list of the data in a row
            for j in range(146):
                data.append(t[i][j])
            d = dict(zip(t.colnames[0:146], data))
            hdr = fits.Header(cards=d)

            xy = WCS(hdr).all_world2pix(self.pos[0], self.pos[1], 1, quiet=True)
            x_cen, y_cen, l, w = t['POST_CEN_X'][i], t['POST_CEN_Y'][i], t['POST_SIZE1'][i]/2., t['POST_SIZE2'][i]/2.
            # Checks to see if xy coordinates of source falls within postcard
            if (xy[0] >= x_cen-l) & (xy[0] <= x_cen+l) & (xy[1] >= y_cen-w) & (xy[1] <= y_cen+w):
                if in_file[0]==None:
                    in_file[0]=i
                else:
                    in_file.append(i)
                # If more than one postcard is found for a single source, choose the postcard where the
                # source is closer to the center
                if len(in_file) > 1:
                    dist1 = np.sqrt( (xy[0]-t['POST_CENX'][in_files[0]])**2 + (xy[1]-t['POST_CENY'][in_files[0]])**2  )
                    dist2 = np.sqrt( (xy[0]-t['POST_CENX'][in_files[1]])**2 + (xy[1]-t['POST_CENY'][in_files[1]])**2  )
                    if dist1 >= dist2:
                        in_file[0]=in_file[1]
                    else:
                        in_file[0]=in_file[0]
        # Returns postcard filename & postcard header
        if in_file[0]==None:
            print("Sorry! We don't have a postcard for you. Please double check your source has been observed by TESS")
            return False, False
        else:
            return t['POST_FILE'][in_file[0]], t[in_file[0]]


    def get_pointing(self, header=None, postcard=None):
        """ Gets the pointing model from the website """
        if postcard == None:
            postcard = header['POSTCARD']
        self.camera, self.chip = postcard[11:12], postcard[13:14]
        print(postcard)
        self.sector = postcard[9:10]
        pm_link = urllib.request.urlopen('http://jet.uchicago.edu/tess_postcards/pointingModel_{}_{}-{}.txt'.format(
                self.sector, self.camera, self.chip))
        pm = pm_link.read().decode('utf-8')
        pm = Table.read(pm, format='ascii.basic')
        return pm


    def get_header(self, postcard=None):
        """ Gets postcard header from the website """
        post_link = urllib.request.urlopen('http://jet.uchicago.edu/tess_postcards/postcard.txt')
        post  = post_link.read().decode('utf-8')
        post  = Table.read(post, format='ascii.basic')
        if postcard==None:
            return post
        else:
            where = np.where(post['POST_FILE']==postcard)[0]
            row   = Row(post, where[0])
            data = []
            for i in range(147):
                data.append(row[i])
            d = dict(zip(row.colnames[0:146], data))
            hdr = fits.Header(cards=d)
            return hdr


    def individual_tpf(self, output_fn=None):
        """
        Creates a FITS file for a given source that includes:
            Extension[0] = header
            Extension[1] = (9x9xn) TPF, where n is the number of cadences in an observing run
            Extension[2] = (3 x n) time, raw flux, systematics corrected flux
        """

        def init_shift(xy):
            """ Offsets (x,y) coords of source by pointing model """
            theta, delX, delY = self.pointing[0]['medT'], self.pointing[0]['medX'], self.pointing[0]['medY']

            x = xy[0]*np.cos(theta) - xy[1]*np.sin(theta) + delX
            y = xy[0]*np.sin(theta) + xy[1]*np.cos(theta) + delY

            return np.array([x,y])

        def centering_shift(tpf):
            """ Creates an additional shift to put source at (4,4) of TPF file """
            """                  Returns: required pixel shift                 """
            xdim, ydim = len(tpf[0][0]), len(tpf[0][1])
            center = quadratic_2d(tpf[0])
            shift = [int(round(xdim/2-center[1])), int(round(xdim/2-center[0]))]

            return shift

        if self.tic != None:
            tic_id, pos, tmag = find_sources.tic_pos_by_ID(self)
            self.pos = pos
            table = find_sources.find_by_position(self)
        elif self.gaia != None:
            table = find_sources.gaia_pos_by_ID(self)
            self.pos = [table['ra'].data[0], table['dec'].data[0]]
            table = find_sources.find_by_position(self)
        elif self.pos != None:
            table = find_sources.find_by_position(self)

        self.pos = [table['RA'].data[0], table['Dec'].data[0]]

        postcard, card_info = self.find_postcard()
        if postcard==False:
            return

        # Extracts sector, camera, and chip from postcard filename
        self.pointing = self.get_pointing(postcard=postcard)

        data = []
        for i in range(147):
            data.append(card_info[i])
        d = dict(zip(card_info.colnames[0:146], data))
        hdr = fits.Header(cards=d)

        # Read in FFI pixel coords of target from RA & dec
        xy = WCS(hdr).all_world2pix(self.pos[0], self.pos[1], 1)

        # Apply initial shift from pointing model
        xy = init_shift(xy)

        # Check if postcard has been downloaded
        # If not, download it
        self.post_dir = self.root_dir + '/postcards/'
        self.post_url = 'http://jet.uchicago.edu/tess_postcards/'

        if os.path.isdir(self.post_dir) == False:
            os.system('cd {} && mkdir postcards'.format(self.root_dir))
        if Path(self.post_dir+postcard).is_file() == False:
            print("*************")
            print("You don't have the postcard your source is on. Here: we'll download it for you!")
            print("*************")
            os.system('cd {} && curl -O -L {}'.format(self.post_dir, self.post_url+postcard) )

        # Read in postcard info
        post_fits = fits.open(self.post_dir+postcard)[0].data
        time_info = fits.open(self.post_dir+postcard)[1].data
        time = (time_info[1]+time_info[0])/2. + time_info[2]

        # Calculate pixel distance between center of postcard and target in FFI pixel coords
        delta_pix = np.array([xy[0] - card_info['POST_CEN_X'], xy[1] - card_info['POST_CEN_Y']])

        # Apply shift to coordinates for center of tpf
        newX = int(np.ceil(card_info['POST_SIZE2']/2. + delta_pix[1]))
        newY = int(np.ceil(card_info['POST_SIZE1']/2. + delta_pix[0]))

        # Define tpf as region of postcard around target
        tpf = post_fits[:,newX-6:newX+7, newY-6:newY+7]

        '''
        # Grab centroid of brightest object in tpf, force to center
        # MAYBE NOT A GOOD IDEA
        xy_new = centering_shift(tpf)
        X, Y = int(newX-xy_new[1]), int(newY-xy_new[0])
        tpf = post_fits[:,X-4:X+5, Y-4:Y+5]
        '''

        radius, shape, lc, uncorrLC = self.aperture_fitting(tpf=tpf)

        if shape == 0:
            shape = 'circle'
        else:
            shape = 'rectangle'

        lcData = [time, np.array(uncorrLC), np.array(lc)]

        # Additional header information
        hdr.append(('COMMENT', '***********************'))
        hdr.append(('COMMENT', '*     ELLIE INFO      *'))
        hdr.append(('COMMENT', '***********************'))
        hdr.append(('AUTHOR' , 'Adina D. Feinstein'))
        hdr.append(('VERSION', '1.0'))
        hdr.append(('GITHUB' , 'https://github.com/afeinstein20/ELLIE'))
        hdr.append(('CREATED', strftime('%Y-%m-%d'),
                    'ELLIE file creation date (YYY-MM-DD)'))
        hdr.append(('POSTCARD', postcard, 'Postcard Filename'))
        hdr.append(('AP_SHAPE', shape))
        hdr.append(('AP_RAD', radius))
        if self.tic !=None:
            hdr.append(('TMAG', float(tmag[0])))
        xy = WCS(hdr).all_world2pix(self.pos[0], self.pos[1], 1)

        hdr.append(('CENTER_X', float(xy[0])))
        hdr.append(('CENTER_Y', float(xy[1])))
        hdr.append(('CEN_RA', float(self.pos[0])))
        hdr.append(('CEN_DEC', float(self.pos[1])))

        # Saves to FITS file
        hdu1 = fits.PrimaryHDU(header=hdr)
        hdu2 = fits.ImageHDU()
        hdu1.data = tpf
        hdu2.data = lcData
        new_hdu = fits.HDUList([hdu1, hdu2])

        if self.tic != None:
            fn = self.tic_fn
        elif self.gaia != None:
            fn = self.gaia_fn
        elif self.pos != None:
            fn = 'hlsp_ellie_tess_ffi_CLOSEST_TIC{}_v1_lc.fits'.format(int(table['TIC_ID']))

        if output_fn==None:
            new_hdu.writeto(fn, overwrite=True)
        else:
            fn = output_fn
            new_hdu.writeto(fn, overwrite=True)
        self.target_fn = fn
        return fn


    def plot(self):
        """
        Makes a simple plot of the light curve and image of TPF
        """
        hdu = fits.open(self.target_fn)
        tpf = hdu[0].data
        lc  = hdu[1].data
        fig  = plt.figure(figsize=(18,5))
        spec = gridspec.GridSpec(ncols=3, nrows=1)
        ax2   = fig.add_subplot(spec[0,2])
        ax1  = fig.add_subplot(spec[0, 0:2])

        ax2.imshow(tpf[0], origin='lower')
        ax2.set_xlabel('Pixel Column Number')
        ax2.set_ylabel('Pixel Row Number')
        ax1.plot(lc[0], lc[1], 'k', label='Uncorrected')
        ax1.plot(lc[0], lc[2], 'r', label='Corrected')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Normalized Flux')
        plt.show()
        return


    def aperture_fitting(self, tpf=None):
        """
        Finds the "best" (i.e. the smallest std) light curve for a range of
        sizes and shapes
        Parameters
        ----------
            sources: list of sources to find apertures for
        """

        def centroidOffset(tpf, file_cen):
            """ Finds offset between center of TPF and centroid of first cadence """
            tpf_com = Cutout2D(tpf[0], position=(len(tpf[0])/2, len(tpf[0])/2), size=(4,4))
            com = ndimage.measurements.center_of_mass(tpf_com.data.T - np.median(tpf_com.data))
            return len(tpf_com.data)/2-com[0], len(tpf_com.data)/2-com[1]

        def aperture(r, pos):
            """ Creates circular & rectangular apertures of given size """
            circ = CircularAperture(pos, r)
            rect = RectangularAperture(pos, r, r, 0.0)
            return circ, rect

        def findLC(x_point, y_point):
            nonlocal tpf
            """ Finds the lightcurve with the least noise by minimizing std """
            r_list = np.arange(1.5, 3.5, 0.5)
            matrix = np.zeros( (len(r_list), 2, len(tpf)) )
            system = np.zeros( (len(r_list), 2, len(tpf)) )
            sigma  = np.zeros( (len(r_list), 2) )

            for i in range(len(r_list)):
                for j in range(len(tpf)):
                    pos = (x_point[j], y_point[j])
                    circ, rect = aperture(r_list[i], pos)
                    # Completes aperture sums for each tpf.flux and each aperture shape
                    matrix[i][0][j] = aperture_photometry(tpf[j], circ)['aperture_sum'].data[0]
                    matrix[i][1][j] = aperture_photometry(tpf[j], rect)['aperture_sum'].data[0]
                matrix[i][0] = matrix[i][0] / np.nanmedian(matrix[i][0])
                matrix[i][1] = matrix[i][1] / np.nanmedian(matrix[i][1])

            print("*************")
            print("Please hold while we do some systematics corrections.")
            print("*************")
            # Creates a complete, systematics corrected light curve for each aperture
            for i in range(len(r_list)):
                lc_circ = self.system_corr(matrix[i][0], x, y, jitter=True, roll=True)
                system[i][0] = lc_circ
                sigma[i][0] = np.std(lc_circ)
                lc_rect = self.system_corr(matrix[i][1], x, y, jitter=True, roll=True)
                system[i][1] = lc_rect
                sigma[i][1] = np.std(lc_rect)

            best = np.where(sigma==np.min(sigma))
            r_ind, s_ind = best[0][0], best[1][0]
            lc_best = system[r_ind][s_ind]
            return r_list[r_ind], s_ind, lc_best, matrix[r_ind][s_ind]

        file_cen = len(tpf[0])/2.
        initParams = self.pointing[0]
        theta, delX, delY = self.pointing['medT'].data, self.pointing['medX'].data, self.pointing['medY'].data
        startX, startY = centroidOffset(tpf, file_cen)

        x, y = [], []
        for i in range(len(theta)):
            x.append( startX*np.cos(theta[i]) - startY*np.sin(theta[i]) + delX[i] )
            y.append( startX*np.sin(theta[i]) + startY*np.cos(theta[i]) + delY[i] )
        x, y = np.array(x), np.array(y)

        print("*************")
        print("We're doing our best to find the ideal aperture shape & size for your source.")
        print("*************")
        radius, shape, lc, uncorr = findLC(x, y)

        return radius, shape, lc, uncorr


    def custom_aperture(self, shape=None, r=0.0, l=0.0, w=0.0, t=0.0, pointing=True,
                        jitter=True, roll=True, input_fn=None, pos=[]):
        """
        Allows the user to input their own aperture of a given shape (either 'circle' or
            'rectangle' are accepted) of a given size {radius of circle: r, length of rectangle: l,
            width of rectangle: w, rotation of rectangle: t}
        The user can have the aperture not follow the pointing model by setting pointing=False
        The user can determine which kinds of corrections would like to be applied to their light curve
            jitter & roll are automatically set to True
        Pos is the position given in pixel space
        """
        def create_ap(pos):
            """ Defines the custom aperture, as inputted by the user """
            nonlocal shape, r, l, w, t
            if shape=='circle':
                return CircularAperture(pos, r)
            elif shape=='rectangle':
                return RectangularAperture(pos, l, w, t)
            else:
                print("Shape of aperture not recognized. Please input: circle or rectangle")
            return

        def cust_lc(center, x=None, y=None):
            nonlocal tpf
            """ Creates the light curve for both cases (if pointing model is on or off) """
            lc = []
            for i in range(len(tpf)):
                if pointing==True:
                    ap = create_ap((x[i], y[i]))
                else:
                    ap = create_ap(center)
                lc.append(aperture_photometry(tpf[i], ap)['aperture_sum'].data[0])
            lc = np.array(lc)/np.nanmedian(lc)
            return lc

        # Checks for all of the correct inputs to create an aperture
        if shape=='circle' and r==0.0:
            print("Please input a radius of your aperture when calling custom_aperture(shape='circle', r=#)")
            return
        if shape=='rectangle' and (l==0.0 or w==0.0):
            print("You are missing a dimension of your rectangular aperture. Please set custom_aperture(shape='rectangle', l=#, w=#)")
            return

        if self.tic != None:
            hdu = fits.open(self.tic_fn)
        elif self.gaia != None:
            hdu = fits.open(self.gaia_fn)
        else:
            hdu = fits.open(input_fn)

        tpf = hdu[0].data

        if len(pos) < 2:
            center = [4,4]
        else:
            center = pos
        # Grabs the pointing model if the user has set pointing==True
        if pointing==True:
            pm  = self.get_pointing(header = hdu[0].header)
            x = center[0]*np.cos(pm['medT'].data) - center[1]*np.sin(pm['medT'].data) + pm['medX'].data
            y = center[0]*np.sin(pm['medT'].data) + center[1]*np.cos(pm['medT'].data) + pm['medY'].data

        x, y = x-np.median(x)+center[0], y-np.median(y)+center[1]
        lc = cust_lc(center, x, y)

        if pointing==False and roll==True:
            print("Sorry, our roll correction, lightkurve.SFFCorrector, only works when you use the pointing model.\nFor now, we're turning roll corrections off.")
            roll=False

        lc = self.system_corr(lc, x, y, jitter=jitter, roll=roll)
        return lc

    def system_corr(self, lc, x_pos, y_pos, jitter=False, roll=False):
        """
        Allows for systematics correction of a given light curve
        Parameters
        ----------
            lc: np.array() of light curve values
            x_pos: np.array() of x positions for the centroid
            y_posL np.array() of y positions for the centroid
        """
        def jitter_corr(lc, x_pos, y_pos):
            x_pos, y_pos = np.array(x_pos), np.array(y_pos)

            def parabola(params, x, y, f_obs, y_err):
                c1, c2, c3, c4, c5 = params
                f_corr = f_obs * (c1 + c2*(x-2.5) + c3*(x-2.5)**2 + c4*(y-2.5) + c5*(y-2.5)**2)
                return np.sum( ((1-f_corr)/y_err)**2)

            # Masks out anything >= 2.5 sigma above the mean
            mask = np.ones(len(lc), dtype=bool)
            for i in range(5):
                lc_new = []
                std_mask = np.std(lc[mask])

                inds = np.where(lc <= np.mean(lc)-2.5*std_mask)
                y_err = np.ones(len(lc))**np.std(lc)
                for j in inds:
                    y_err[j] = np.inf
                    mask[j]  = False

                if i == 0:
                    initGuess = [3, 3, 3, 3, 3]
                else:
                    initGuess = test.x

                bnds = ((-15.,15.), (-15.,15.), (-15.,15.), (-15.,15.), (-15.,15.))
                test = minimize(parabola, initGuess, args=(x_pos, y_pos, lc, y_err), bounds=bnds)
                c1, c2, c3, c4, c5 = test.x
                lc_new = lc * (c1 + c2*(x_pos-2.5) + c3*(x_pos-2.5)**2 + c4*(y_pos-2.5) + c5*(y_pos-2.5)**2)
            return lc_new


        def rotation_corr(lc, x_pos, y_pos):
            """ Corrects for spacecraft roll using Lightkurve """
            time = np.arange(0, len(lc), 1)
            sff = SFFCorrector()
            x_pos, y_pos = np.array(x_pos), np.array(y_pos)
            lc_corrected = sff.correct(time, lc, x_pos, y_pos, niters=1,
                                       windows=1, polyorder=5)
            return lc_corrected.flux

        if jitter==True and roll==True:
            newlc = jitter_corr(lc, x_pos, y_pos)
            newlc = rotation_corr(newlc, x_pos, y_pos)
        elif jitter==True and roll==False:
            newlc = jitter_corr(lc, x_pos, y_pos)
        elif jitter==False and roll==True:
            newlc = rotation_corr(lc, x_pos, y_pos)
        elif jitter==False and roll==False:
            newlc = lc
        return newlc



##########################
##########################
##########################
class visualize(data_products, find_sources):
    """
    The main interface for creating figures, movies, and interactive plots
    Allows the user to have a grand ole time playing with their data!

    Args:
        tpf: A FITS file that contains stacked cadences for a single source
    """

    def __init__(self, tic=None, gaia=None, input_fn=None, **kwargs):
        """ USER INPUT """
        # Get ID out of file
        self.tic  = tic
        self.gaia = gaia
        if self.tic != None:
            self.fn   = 'hlsp_ellie_tess_ffi_{}_v1_lc.fits'.format(self.tic)
        elif self.gaia != None:
            self.fn   = 'hlsp_ellie_tess_ffi_GAIA{}_v1_lc.fits'.format(self.gaia)
        else:
            self.fn = input_fn


    def tpf_movie(self, output_fn=None, cbar=True, aperture=False, com=False, plot_lc=False, **kwargs):
        """
        This function allows the user to create a TPF movie
        Parameters
        ----------
            cmap: Allows the user to choose the color map for their movie
                  (Defaults to the only acceptable colormap)
            cbar: Allows the user to decide if they want a colorbar for scale
                  (Defaults to True)
            aperture: Allows the user to decide if they want the aperture on
                      their movie (Defaults to False)
            com: Allows the user to decide if they want to see the center of
                 mass of the target (Defaults to False)
            lc: Allows the user to plot the light curve and movement along light curve
                with TPF movie (Defaults to False)
        Returns
        ----------
            Creates an MP4 file
        """
        hdu = fits.open(self.fn)
        tp  = hdu[0].data
        lc_dat = hdu[1].data
        time = lc_dat[0]
        lc = lc_dat[2]
        lc = lc/np.nanmedian(lc)

        if com==True or aperture==True:
            pm = data_products.get_pointing(self, header=hdu[0].header)
            x, y = [], []
            for i in range(len(tp)):
                tp_com = Cutout2D(tp[i], position=(4,4), size=(4,4))
                tp_com = tp_com.data
                centroid = ndimage.measurements.center_of_mass(tp_com.T - np.median(tp_com))
                x.append(centroid[0]+2.)
                y.append(centroid[1]+2.)

        if 'vmax' not in kwargs:
            kwargs['vmax'] = np.max(tp[0])
        if 'vmin' not in kwargs:
            kwargs['vmin'] = np.min(tp[0])

        def animate(i):
            nonlocal line, x, y, scats, line, ps
            ax.imshow(tp[i], origin='lower', **kwargs)
            # Plots motion of COM when the user wants
            if com==True:
                for scat in scats:
                    scat.remove()
                scats = []
                scats.append(ax.scatter(x[i], y[i], s=16, c='k'))

            # Plots aperture around source when the user wants
            if aperture==True:
                for c in ps:
                    c.remove()
                ps=[]
                circleShape = patches.Circle((x[i],y[i]), 1.5, fill=False, alpha=0.4)
                p = PatchCollection([circleShape], alpha=0.4)
                p.set_array(np.array([0]))
                p.set_edgecolor('face')
                ps.append(ax.add_collection(p))

            # Plots moving point along light curve when the user wants
            if plot_lc==True:
                for l in line:
                    l.remove()
                line = []
                line.append(ax1.scatter(time[i], lc[i], s=20, c='r'))

            # Updates the frame number
            time_text.set_text('Frame {}'.format(i))


        line, scats, ps = [],[], []
        if plot_lc==True:
            fig  = plt.figure(figsize=(18,5))
            spec = gridspec.GridSpec(ncols=3, nrows=1)
            ax   = fig.add_subplot(spec[0,2])
            ax1  = fig.add_subplot(spec[0, 0:2])
            ax1.plot(time, lc, 'k')
            ax1.set_ylabel('Normalized Flux')
            ax1.set_xlabel('Time (Days)')
            ax1.set_xlim([np.min(time)-0.05, np.max(time)+0.05])
            ax1.set_ylim([np.min(lc)-0.05, np.max(lc)+0.05])

        elif plot_lc==False:
            fig  = plt.figure()
            spec = gridspec.GridSpec(ncols=1, nrows=1)
            ax   = fig.add_subplot(spec[0,0])

        # Writes frame number on TPF movie
        time_text = ax.text(5.5, -0.25, '', color='white', fontweight='bold')
        time_text.set_text('')

        # Allows TPF movie to be saved as mp4
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='Adina Feinstein'), bitrate=1800)

        ani = animation.FuncAnimation(fig, animate, frames=len(tp))

        if cbar==True:
            plt.colorbar(ax.imshow(tp[0], **kwargs), ax=ax)

        if output_fn == None:
            if self.tic != None:
                self.output_fn = '{}.mp4'.format(self.tic)
                ax.set_title('{}'.format(self.tic), fontweight='bold')
            elif self.gaia != None:
                self.output_fn = '{}.mp4'.format(self.gaia)
                ax.set_title('{}'.format(self.gaia), fontweight='bold')

        plt.tight_layout()
        return ani



    def click_aperture(self, corrected=False):
        """
        Allows the user to click specific pixels they want to create
        a lightcurve for
        """
        def click_pixels():
            nonlocal tpf
            """ Creates a rectangle over a pixel when that pixel is clicked """
            coords, rectList = [], []

            fig, ax = plt.subplots()
            ax.imshow(tpf[0], origin='lower')

            def onclick(event):
                """ Update figure canvas """
                nonlocal coords, rectList
                x, y = int(np.round(event.xdata,0)), int(np.round(event.ydata,0))
                # Highlights pixel
                rect = Rectangle((x-0.5, y-0.5), 1.0, 1.0)
                rect.set_color('white')
                rect.set_alpha(0.4)
                # Adds pixel if not previously clicked
                if [x,y] not in coords:
                    coords.append([x,y])
                    rectList.append(rect)
                    ax.add_patch(rect)
                fig.canvas.draw()
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()
            plt.close()
            return coords, rectList

        def check_pixels():
            nonlocal tpf, coords, rectList
            """ Presents a figure for the user to approve of selected pixels """
            fig, ax = plt.subplots(1)
            ax.imshow(tpf[0], origin='lower')

            # Recreates patches for confirmation
            for i in range(len(coords)):
                x, y = coords[i][0], coords[i][1]
                rect = Rectangle((x-0.5, y-0.5), 1.0, 1.0)
                rect.set_color('red')
                rect.set_alpha(0.4)
                ax.add_patch(rect)

            # Make Buttons
            plt.text(-3.5, 5.5, 'Are you happy\nwith this\naperture?', fontsize=8)
            axRadio  = plt.axes([0.05, 0.45, 0.10, 0.15])
            butRadio = RadioButtons(axRadio, ('Yes', 'No'), activecolor='red')
            good=True

            # Checks if user is happy with custom aperture
            def get_status(value):
                nonlocal good
                if value == 'Yes':
                    good=True
                else:
                    good=False

            butRadio.on_clicked(get_status)
            plt.show()
            return good

        hdu = fits.open(self.fn)
        tpf = hdu[0].data
        coords, rectList = click_pixels()
        check = check_pixels()

        custlc = []
        if check==True:
            for f in range(len(tpf)):
                cadence = []
                for i in range(len(coords)):
                    cadence.append(tpf[f][coords[i][0], coords[i][1]])
                custlc.append(np.sum(cadence))
            custlc = np.array(custlc) / np.nanmedian(custlc)

        ### do the corrections if asked ###

            return custlc

        else:
            self.click_aperture()


    def mark_gaia(self):
        """
        Allows the user to mark other Gaia sources within a given TPF
        Click on the x's to reveal the source's ID and Gmag
        Also cross-matches with TIC and identifies sources in there as well
        """
        def find_gaia_sources(header, hdr):
            """ Compeltes a cone search around the center of the TPF for Gaia sources """
            pos = [header['CEN_RA'], header['CEN_DEC']]
            l   = find_sources(pos=pos)
            sources = l.cone_search(r=0.5, service='Mast.Catalogs.GaiaDR2.Cone')
            xy = WCS(hdr).all_world2pix(sources['ra'], sources['dec'], 1)
            return xy, sources['source_id'], sources['phot_g_mean_mag'], sources['ra'], sources['dec']

        def pointingCorr(xy, header):
            """ Corrects (x,y) coordinates based on pointing model """
            pm = data_products.get_pointing(self, header=header)
            shift = pm[0]
            x = xy[0]*np.cos(shift[0]) - xy[1]*np.sin(shift[0]) + shift[1]
            y = xy[0]*np.sin(shift[0]) + xy[1]*np.cos(shift[0]) + shift[2]
            return np.array([x,y])

        def in_tpf(xy, gaiaXY, gaiaID, gaiaMAG, gaiaRA, gaiaDEC):
            """ Pushes the gaia sources to the appropriate place in the TPF """
            gaiaX, gaiaY = gaiaXY[0]-xy[0]+4, gaiaXY[1]-xy[1]+5
            inds = np.where( (gaiaX >= -0.5) & (gaiaX <= 8.5) &
                             (gaiaY >= -0.5) & (gaiaY <= 8.5) & (gaiaMAG <= 16.5))
            return [gaiaX[inds], gaiaY[inds]], gaiaID[inds], gaiaMAG[inds], gaiaRA[inds], gaiaDEC[inds]

        def create_labels(gaia_id):
            ticLabel, tmagLabel = np.zeros(len(gaia_id.data)), np.zeros(len(gaia_id.data))
            crossTable = find_sources.crossmatch_multi_to_tic(self, list=gaia_id.data)
            for i in range(len(gaia_id.data)):
                row = crossTable[i]
                if row['separation'] <= 1.2 and row['Gmag'] <= 16.5:
                    ticLabel[i]  = row['TIC_ID']
                    tmagLabel[i] = row['Tmag']
            return ticLabel, tmagLabel


#        output_file("gaia_hover.html")

        hdu = fits.open(self.fn)
        tpf = hdu[0].data[0]
        header = hdu[0].header
        center = [header['CENTER_X'],  header['CENTER_Y']]

        hdr = data_products.get_header(self, postcard=header['POSTCARD'])
        # Finds & corrects Gaia sources
        gaia_xy, gaia_id, gaia_gmag, gaia_ra, gaia_dec = find_gaia_sources(header, hdr)
        gaia_xy = pointingCorr(gaia_xy, header)
        gaia_xy, gaia_id, gaia_gmag, gaia_ra, gaia_dec = in_tpf(center, gaia_xy, gaia_id, gaia_gmag, gaia_ra, gaia_dec)

        # Crossmatches Gaia -> TIC
        ticLabel, tmagLabel = create_labels(gaia_id)

        hover = HoverTool()
        hover.tooltips = [
            ("TIC ID", "test"),
            ("T_mag", "test"),
            ("GAIA ID", "test"),
            ("G_mag", "test")
            ]

        xyrange=(-0.5,8.5)
        if self.tic != None:
            p = figure(x_range=xyrange, y_range=xyrange, toolbar_location="above",
                       title='TIC {}'.format(self.tic), plot_width=500, plot_height=450)
        elif self.gaia!= None:
            p = figure(x_range=xyrange, y_range=xyrange, toolbar_location="above",
                       title='Gaia {}'.format(self.gaia), plot_width=500, plot_height=450)

        color_mapper = LinearColorMapper(palette='Viridis256', low=np.min(tpf), high=np.max(tpf))
        tpf_img = p.image(image=[tpf], x=-0.5, y=-0.5, dw=9, dh=9, color_mapper=color_mapper)
        p.xaxis.ticker = np.arange(0,9,1)
        p.yaxis.ticker = np.arange(0,9,1)

        x_overrides, y_overrides = {}, {}
        x_list = np.arange(int(center[0]-4), int(center[0]+5), 1)
        y_list = np.arange(int(center[1]-4), int(center[1]+5), 1)
        for i in np.arange(0,9,1):
            ind = str(i)
            x_overrides[ind] = str(x_list[i])
            y_overrides[ind] = str(y_list[i])

        p.xaxis.major_label_overrides = x_overrides
        p.yaxis.major_label_overrides = y_overrides
        p.xaxis.axis_label = 'Pixel Column Number'
        p.yaxis.axis_label = 'Pixel Row Number'

        color_bar = ColorBar(color_mapper=color_mapper, location=(0,0), border_line_color=None,
                             ticker=BasicTicker(), title='Intensity', title_text_align='left')
        p.add_layout(color_bar, 'right')

        source=ColumnDataSource(data=dict(x=gaia_xy[0], y=gaia_xy[1],
                                          tic=ticLabel, tmag=tmagLabel,
                                          gaia=gaia_id, gmag=gaia_gmag,
                                          ra=gaia_ra  , dec=gaia_dec))


        s = p.circle('x', 'y', size=8, source=source, line_color=None,
                     selection_color='red', nonselection_fill_alpha=0.0, nonselection_line_alpha=0.0,
                     nonselection_line_color=None, fill_color='black', hover_alpha=0.9, hover_line_color='white')
        p.add_tools(HoverTool( tooltips=[("TIC Source", "@tic"), ("Tmag", "@tmag"),
                                         ("Gaia Source", "@gaia"), ("Gmag", "@gmag"),
                                         ("RA", "@ra"), ("Dec", "@dec")],
                               renderers=[s], mode='mouse', point_policy="snap_to_data"))

        output_notebook()

#        show(p)
        return p

#        try:
#            push_notebook()
#        except AttributeError:
#            print("Because you're not working in a Jupyter Notebook, we've saved this figure as a .html for you. Please check your files now for the figure.")
#            output_file("markGaia.html")
