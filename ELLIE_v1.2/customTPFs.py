import os, sys, re, json, time
import requests
from bs4 import BeautifulSoup
import numpy as np
import matplotlib

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
#from astroquery.gaia import Gaia
from astroquery.mast import Catalogs
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import matplotlib.pyplot as plt

import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec

from photutils import CircularAperture, RectangularAperture, aperture_photometry
from lightkurve import KeplerTargetPixelFile as ktpf
from lightkurve import SFFCorrector
from scipy import ndimage
from scipy.optimize import minimize
import collections

try: # Python 3.x
    from urllib.parse import quote as urlencode
    from urllib.request import urlretrieve
    import http.client as httplib
except ImportError: # Python 2.x
    from urllib import pathname2url as urlencode
    from urllib import urlretrieve
    import httplib

class custom_tpf:
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
            source_id, ra, dec, gmag, pmra, pmdec, parallax
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
        return table['source_id'].data, [table['ra'].data[0], table['dec'].data[0]], table['phot_g_mean_mag'].data, table['pmra'].data[0], table['pmdec'].data[0], table['parallax'].data[0]


    def tic_pos_by_ID(self, multiSource=None):
        """
        Finds the RA,Dec for a given TIC source_id
        Parameters
        ----------
            multiSource: used when user passes in a file of TIC IDs to crossmatch
        Returns
        ----------
            source_id, ra, dec, tmag 
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
        r = 0.001
        t = self.initialize_table()
        for s in sources:
            tID, pos, tmag = self.tic_pos_by_ID(s)
            gaia = self.crossmatch_by_position(r, service, pos)
            pos[0], pos[1] = pos[0]*u.deg, pos[1]*u.deg
            separation = self.crossmatch_distance(pos, [gaia['MatchRA'], gaia['MatchDEC']])
            t.add_row([gaia['MatchID'], s, (pos[0]).to(u.arcsec), (pos[1]).to(u.arcsec), separation, gaia['phot_g_mean_mag'], tmag, gaia['pmra'], gaia['pmdec'], gaia['parallax']])
        t.remove_row(0)
        return t

    
    def crossmatch_multi_to_tic(self):
        """
        Crossmatches file of Gaia IDs to TIC
        Parameters
        ----------
        Returns
        ----------
            table: table of gaia_id, tic_id, ra, dec, delta_pos, gmag, tmag, pmra, pmdec, parallax 
        """
        filename = self.multiFile
        sources  = np.loadtxt(filename, dtype=int)
        service  = 'Mast.Tic.Crossmatch'
        r = 0.01
        t = self.initialize_table()
        for s in sources:
            sID, pos, gmag, pmra, pmdec, parallax = self.gaia_pos_by_ID(s)
            tic = self.crossmatch_by_position(r, serice, pos)
            pos[0], pos[1] = pos[0]*u.deg, pos[1]*u.deg
            separation = self.crossmatch_distance(pos, [tic['MatchRA'], tic['MatchDEC']])
            t.add_row([s, tic['MatchID'], (pos[0]).to(u.arcsec), (pos[1]).to(u.arcsec), separation, gmag, tic['Tmag'], pmra, pmdec, parallax])
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
        if self.multiFile[-3::] == 'csv':
            data = np.loadtxt(self.multiFile, delimiter=',')
        else:
            data = np.loadtxt(self.multiFile)

        columns = ['Gaia_ID', 'TIC_ID', 'RA', 'Dec', 'Gaia_sep', 'TIC_sep', 'Gmag', 'Tmag', 'pmra', 'pmdec', 'parallax']
        t = Table(np.zeros(10), names=columns)
        t['RA'].unit, t['Dec'].unit, t['separation'].unit = u.arcsec, u.arcsec, u.arcsec
        t['pmra'].unit, t['pmdec'].unit = u.mas/u.year, u.mas/u.year

        for i in range(len(data)):
            pos = data[i]
            gaia = self.crossmatch_by_position(0.1, 'Mast.GaiaDR2.Crossmatch', pos)
            tess = self.crossmatch_by_position(0.5, 'Mast.Tic.Crossmatch', pos)
            pos[0], pos[1] = pos[0]*u.deg, pos[1]*u.deg
            gaiaPos = [gaia['MatchRA'], gaia['MatchDEC']]
            sepGaia = self.crossmatch_distance(pos, gaiaPos)
            tessPos = [tess['MatchRA'], tess['MatchDEC']]
            sepTess = self.crossmatch_distance(pos, tessPos)
            
            t.add_row([gaia['source_id'], tess['MatchID'], pos[0], pos[1], sepGaia, sepTess, gaia['phot_g_mean_mag'], 
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

    
    def pointing_model(self):
        
        def sort_by_date(fns, dir):
            """
            Sort FITS files by start date of observation
            Parameters
            ---------- 
                dir: directory where the FITS files are
            Returns
            ---------- 
                fns: sorted filenames by date
            """
            dates = []
            for f in fns:
                header = fits.getheader(dir+f)
                dates.append(header['DATE-OBS'])
                dates, fns = np.sort(np.array([dates, fns]))
            return fns


        def nearest(x, y, x_list, y_list):
            """
            Calculates the distance to the nearest source
            Parameters
            ---------- 
                x: x-coord of source
                y: y-coord of source
                x_list: x-coords of all sources 
                y_list: y-coords of all sources
            Returns
            ---------- 
                dist: distance to closest source
            """
            x_list = np.delete(x_list, np.where(x_list==x))
            y_list = np.delete(y_list, np.where(y_list==y))
            dist   = np.sqrt( (x-x_list)**2 + (y-y_list)**2 )
            return np.min(dist)


        def find_isolated(x, y):
            """
            Find isolated sources in the image where an isolated source is 
            >= 8.0 pixels away from any other source
            Parameters
            ---------- 
                x: a list of x-coords for all sources
                y: a list of y-coords for all sources
            Returns
            ---------- 
                isolated: a list of isolated source indices
            """
            isolated = []
            for i in range(len(x)):
                dist = nearest(x[i], y[i], x, y)
                if dist >= 8.0:
                    isolated.append(i)
            return isolated

        def calc_shift(dir, fns, x, y, corrFile, mast):
            """
            Calculates the deltaX, deltaY, and deltaTheta (rotation) for isolated 
            sources in order to create a pointing model for the entire chip
            Parameters
            ---------- 
                dir: directory FITS files are in
                fns: all FITS filenames
                x  : a list of x-coordinates for isolated sources
                y  : a list of y-coordinates for isolated sources
                corrFile: name of file to write corrections to for each cadence
                mast: data for first cadence (for reference)
            Returns
            ---------- 
            """
            
            # Fits the best deltaX, deltaY, deltaT
            def model(params):
                nonlocal x, y, centroids
                theta, xT, yT = params
                xRot = x*np.cos(theta) - y*np.sin(theta) + xT
                yRot = x*np.sin(theta) + y*np.cos(theta) + yT
                dist = np.sqrt( (xRot-centroids[:,0])**2 + (yRot-centroids[:,1])**2 )
                return np.sum( np.square(dist) )
            fns = [dir+i for i in fns]
            
            # Shifts file to center at (0,0)
            x, y = x-len(mast)/2, y-len(mast[0])/2
            
            with open(corrFile, 'w') as tf:
                tf.write('cadence, medT, medX, medY\n')
                
            matrix = np.zeros((len(x), len(fns), 2))

            for i in range(len(x)):
                tpf = ktpf.from_fits_images(images=fns, position=(x[i], y[i]), size=(6,6))
                #tpf.to_fits(output_fn='{}_tpf.fits'.format(i))
                for j in range(len(fns)):
                    com = ndimage.measurements.center_of_mass(tpf.flux[j].T-np.median(tpf.flux[j])) # subtracts background
                    matrix[i][j][0] = com[0]+x[i]
                    matrix[i][j][1] = com[1]+y[i]
            for i in range(len(x)):
                centroids = matrix[:,i]
                if i == 0:
                    initGuess = [0.001, 0.1, -0.1]
                else:
                    initGuess = solution.x
                bnds = ((-0.08, 0.08), (-5.0, 5.0), (-5.0, 5.0))
                solution = minimize(model, initGuess, method='L-BFGS-B', bounds=bnds, options={'ftol':5e-11,
                                                                                               'gtol':5e-05})
                with open(corrFile, 'a') as tf:
                    tf.write('{}\n'.format(str(i) + ' ' + ' '.join(str(e) for e in success.x)))
            return
        
        filnames = np.array(os.listdir(self.dir))
        fitsInds = np.array([i for i,item in enumerate(filenames) if "fits" in item])
        filenames = sort_by_date(filenames[fitsInds], dir)
        
        mast, header = fits.getdata(self.dir, filenames[0])
        pos = [header['CRVAL1'], header['CRVAL2']]
        dataTable = self.tic_by_contamination(pos, 6*np.sqrt(2), [0.0, 5e-4])
        ra, dec = dataTable['ra'], dataTable['dec']
        xy = WCS(header).all_world2pix(ra,dec,1)

        in_frame = np.where((y>=44.) & (y<len(mast)-45.) & (x>=0.) & (x<len(mast[0])-41.))[0]
        x, y = xy[0][in_frame], xy[1][in_frame]
        
        isolated = find_isolated(x,y)
        x, y = x[isolated], y[isolated]
        calcShift(dir, filenames, x, y, 'pointingModel_{}-{}.txt'.format(self.camera, self.chip), mast)
        return

    
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


    def make_postcard(self):
        """
        Creates a "postcard" of (350,350) pixels and stacks all cadences in this one file
        Allows the user to easily create multiple individual TPFs when sources are in the same region
        Parameters
        ----------
        Returns
        ----------
        """
        fns = np.array(os.listdir(self.dir))
        fns = fns[np.array([i for i,item in enumerate(fns) if 'fits' in item])]
        fns = [dir+i for i in fns]
        
        mast, header = fits.getdata(fns[0], header=True)

        x, y = np.linspace(0, len(mast), 4, dtype=int), np.linspace(0, len(mast[0]), 4, dtype=int)
        x_cens, y_cens = [], []
        
        cat = 'postcard_catalog.txt'
        with open(Cat, 'w') as tf:
            tf.write('filename ra_low ra_up dec_low dec_up\n')
        for i in range(len(x)-1):
            for j in range(len(y)-1):
                 fn = 'postcard_{}-{}.fits'.format(i, j)
                 x_cen = (x[i]+x[i+1]) / 2.
                 y_cen = (y[j]+y[j+1]) / 2.
                 x_cens.append(x_cen)
                 y_cens.append(y_cen)
                 
                 # Converts (RA,Dec) -> (x,y) and creates TPF for postcard
                 radec = WCS(mheader).all_pix2world(x_cen, y_cen, 1)
                 tpf = ktpf.from_fits_images(images=fns, position=(x_cen,y_cen), size=(350,350))
                 tpf.to_fits(output_fn=fn)

                 # Edits header of fits file for additional information
                 fits.setval(fn, 'CEN_X' , value=np.round(x_cen,5))
                 fits.setval(fn, 'CEN_Y' , value=np.round(y_cen,5))
                 fits.setval(fn, 'CEN_RA', value=float(radec[0]))
                 fits.setval(fn, 'CEN_DEC', value=float(radec[1]))

                 # Adds to catalog of postcard information
                 lower = WCS(mheader).all_pix2world(x_cen-350, y_cen-350, 1)
                 upper = WCS(mheader).all_pix2world(x_cen+350, y_cen+350, 1)
                 row = [fn, lower[0], lower[1], upper[0], upper[1]]
                 with open(cat, 'a') as tf:
                     tf.write('{}\n'.format(' '.join(str(e) for e in row)))
        return
    

class create_lightcurve:
    """
    Allows the user to create their own light curve. This includes:
    1. Correcting for systematics & choosing which systematics to correct for (jitter vs. roll)
    2. Choosing a custom aperture for the source
    """
    def __init__(self, id, dir=None, camera=None, chip=None):
        self.id     = id
        self.camera = camera
        self.chip   = chip
        if dir==None:
            self.tpf_file = '{}_tpf.fits'.format(self.id)
        else:
            self.tpf_file = dir + '{}_tpf.fits'.format(self.id)
            

    def custom_aperture(self):
        return


    def lightcurve(self):
        """
        Loads in source TPF and pointing model to create a custom light curve that tracks
        the pointing model/center of mass of the source
        Parameters
        ---------- 
        Returns
        ---------- 
            lc / np.nanmedian(lc): normalized lightcurve
            x_point: x-coordinate of source after pointing model correction
            y_point: y-coordinate of source after pointing model correction
        """
        theta, delX, delY = np.loadtxt('pointingModel_{}-{}.txt'.format(self.camera, self.chip),
                                       skiprows=1, usecls=(1,2,3), unpack=True)
        tpf = ktpf.from_fits(self.tpf_file)
        hdu = fits.open(self.tpf_file)
        
        cen_x, cen_y = len(tpf.flux[0])/2., len(tpf.flux[0])/2.
        
        # Creates estimated center location taking the pointing model into account
        x_point = cen_x * np.cos(np.radians(theta)) - cen_y * np.sin(np.radians(theta)) - delx
        y_point = cen_x * np.sin(np.radians(theta)) + cen_y * np.cos(np.radians(theta)) - dely
        
        # Creates the light curve given different center locations per cadence
        lc = []
        aperture = self.custom_aperture()
        for f in range(len(tpf.flux)):
            pos = [x_point[f], y_point[f]]
            lc.append(aperture_photometry(tpf.flux[f], aperture)['aperture_sum'].data[0])
        return np.array(lc / np.nanmedian(lc)), x_point, y_point

        
    def jitter_correction(self):
        """
        Corrects for systematic errors stemming from jitter of the telescope
        Jitter correction method taken from Knutson et al. (2008) https://arxiv.org/pdf/0709.3984.pdf
        Parameters
        ----------
        Returns
        ----------
            lc: normalized & corrected light curve
        """

        def parabola(params, x, y, f_obs, y_err):
            """ Used to minimize systematics from jitter """
            c1, c2, c3, c4, c5 = params
            f_corr = f_obs * (c1 + c2*(x-2.5) + c3*(x-2.5)**2 + c4*(y-2.5) + c5*(y-2.5)**2)
            return np.sum( ( (1-f_corr)/y_err)**2)


        lc_norm, x_point, y_point = self.lightcurve()
        
        # Masks out anything >= 2.5 sigma above the mean
        mask = np.ones(len(lc), dtype=bool)
        for i in range(5):
            lc_new = []
            std_mask = np.std(lc[mask])
            inds = np.where(lc <= np.mean(lc)-2.5*std_mask)
            y_err = np.ones(len(lc))*np.std(lc)

            for j in inds:
                y_err[j] = np.inf
                mask[j]  = False

            if i == 0:
                initGuess = [3, 3, 3, 3, 3]
            else:
                initGuess = test.x

            bnds = ((-15.,15.), (-15.,15.), (-15.,15.), (-15.,15.), (-15.,15.))
            test = minimize(parabola, initGuess, args=(x_point, y_point, lc, y_err), bounds=bnds)
            
            c1, c2, c3, c4, c5 = test.x
            lc_new = lc * (c1 + c2*(x_point-2.5) + c3*(x_point-2.5)**2 + c4*(y_point-2.5) + c5*(y_point-2.5)**2)
        return lc_new

    
    def roll_correction(self):
        """
        Corrects for light curve systematics associated with roll of the telescope
        Taken from Vanderburg & Johnson (2014) https://arxiv.org/pdf/1408.3853.pdf as implemented by lightkurve 
        ----------
           lc: normalized & corrected light curve
        """
        lc_nrom, x_point, y_point = self.lightcurve()
        time = np.arange(0, len(lc), 1)
        
        sff = SFFCorrector()
        lc_corrected = sff.correct(time, lc, x_point, y_point, niters=1, windows=1, polyorder=5)
        long_term_trend = sff.trend
        return lc_corrected.flux*long_term_trend

        

class visualize:
    """
    The main interface for creating figures, movies, and interactive plots
    Allows the user to have a grand ole time playing with their data!
        
    Args:
        tpf: A FITS file that contains stacked cadences for a single source
    """

    def __init__(self, id, dir=None):
        """ USER INPUT """
        self.id  = id
        if dir==None:
            self.tpf = '{}_tpf.fits'.format(self.id)
            self.lcf = '{}_lc.fits'.format(self.id)
        else:
            self.tpf = dir+'{}_tpf.fits'.format(self.id)
            self.lcf = dir+'{}_lc.fits'.format(self.id)

        try:
            fits.getdata(self.lcf)
        except IOError:
            print('Please input directory FITS files are in.')
            return
        

    def tpf_movie(self, output_fn=None, cmap='viridis', cbar=True, aperture=False, com=False, plot_lc=False):
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
        tp = ktpf.from_fits(self.tpf)
        lc = tp.to_lightcurve()
#        lc = fits.getdata(self.lcf)
        time, lc = lc.time, lc.flux/np.nanmedian(lc.flux)

        cbmax = np.max(tp.flux[0])
        cbmin = np.min(tp.flux[0])

        def animate(i):
            nonlocal line

            ax.imshow(tp.flux[i], origin='lower', cmap=cmap, vmin=cbmin, vmax=cbmax)
            
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

        
        line, scats = [],[]
        if plot_lc==True:
            fig  = plt.figure(figsize=(18,5))
            spec = gridspec.GridSpec(ncols=3, nrows=1)
            ax   = fig.add_subplot(spec[0,2])
            ax1  = fig.add_subplot(spec[0, 0:2])
            ax1.plot(time, lc, 'k')
            ax1.set_ylabel('Normalized Flux')
            ax1.set_xlabel('Time - 2454833 (Days)')
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

        ani = animation.FuncAnimation(fig, animate, frames=len(tp.flux))

        if cbar==True:
            plt.colorbar(plt.imshow(tp.flux[0], cmap=cmap, vmin=cbmin, vmax=cbmax), ax=ax)

        if output_fn == None:
            output_fn = '{}.mp4'.format(self.id)        
            
        plt.tight_layout()
        ax.set_title('{}'.format(self.id), fontweight='bold')
#        plt.show()
        return ani
