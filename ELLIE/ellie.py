import os, sys, re, json, time
import requests
from bs4 import BeautifulSoup
import numpy as np
import matplotlib

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astroquery.mast import Catalogs
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table, Column
from astropy.wcs import WCS

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.widgets import RadioButtons

import mplcursors

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
        print("This is the position I'm using: ", pos)
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
#        gaiaData = Catalogs.query_region(catalog="Gaia", ID=source, version=2)
#        return gaiaData['ID'].data, [gaiaData['ra'].data[0], gaiaData['dec'].data[0]], gaiaData['phot_g_mean_mag'].data
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
            filename = self.multiFile
            sources  = np.loadtxt(filename, dtype=int)
        else:
            sources = list
        service  = 'Mast.Tic.Crossmatch'
        r = 0.05
        t = self.initialize_table()
        for s in sources:
            self.gaia = s
            table = self.gaia_pos_by_ID()
            sID, pos, gmag = table['source_id'].data, [table['ra'].data[0], table['dec'].data[0]], table['phot_g_mean_mag'].data
            pmra, pmdec, parallax = table['pmra'].data, table['pmdec'].data, table['parallax'].data
            tic = self.crossmatch_by_position(r, service, pos)
            pos[0], pos[1] = pos[0]*u.deg, pos[1]*u.deg
            separation = self.crossmatch_distance(pos, [tic['MatchRA'], tic['MatchDEC']])
            min = separation.argmin()
            row = tic[min]
            t.add_row([sID, row['MatchID'], (pos[0]).to(u.arcsec), (pos[1]).to(u.arcsec), separation[min], gmag, row['Tmag'], pmra, pmdec, parallax])
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
            gaia = self.crossmatch_by_position(0.005, 'Mast.GaiaDR2.Crossmatch', pos)[0]
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
    def __init__(self, id=None, dir=None, camera=None, chip=None, sector=None):
        self.id     = id
        self.camera = camera
        self.chip   = chip
        self.dir    = dir
        self.sector = sector
        self.corrFile = 'pointingModel_{}_{}-{}.txt'.format(sector, camera, chip)
        if dir==None:
            self.tpf_file = '{}_tpf.fits'.format(self.id)
        else:
            self.tpf_file = dir + '{}_tpf.fits'.format(self.id)
            

    def sort_by_date(self):
        """ Sorts FITS files by start date of observation """
        fns = np.array(os.listdir(self.dir))
        fitsInds = np.array([i for i,item in enumerate(fns) if 'fits' in item])
        fns = fns[fitsInds]
        dates = []
        for f in fns:
            mast, header = fits.getdata(self.dir+f, header=True)
            dates.append(header['DATE-OBS'])
        dates, fns = np.sort(np.array([dates, fns]))
        return fns


    def pointing_model(self):
        """
        Creates the pointing model for a given camera and chip across all cadences
        """
        
        def pixel_cone(header, r, contam):
            """ Completes a cone search around center of FITS file """
            pos = [header['CRVAL1'], header['CRVAL2']]
            data = find_sources.tic_by_contamination(find_sources(), pos, r, contam)
            ra, dec = data['ra'], data['dec']
            return WCS(header).all_world2pix(ra, dec, 1), data['ID']
        

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
                theta = np.radians(theta)

                xRot = xy[:,0]*np.cos(theta) - xy[:,1]*np.sin(theta)+xT
                yRot = xy[:,0]*np.sin(theta) + xy[:,1]*np.cos(theta)+yT

                dist = np.sqrt((xRot-centroids[:,0])**2 + (yRot-centroids[:,1])**2)
                return np.sum(np.square(dist))

            fns = np.array([self.dir+i for i in fns])
            x_cen, y_cen = len(mast)/2., len(mast[0])/2.
            xy = np.zeros((len(x),2))
            matrix = np.zeros((len(x), len(fns), 2))

            with open(self.corrFile, 'w') as tf:
                tf.write('cadence medT medX medY\n')

            for i in range(len(x)):
                xy[i][0], xy[i][1] = x[i]-x_cen, y[i]-y_cen
                tpf = ktpf.from_fits_images(images=fns, position=(x[i],y[i]), size=(6,6))
                for j in range(len(fns)):
                    com = ndimage.measurements.center_of_mass(tpf.flux[j].T-np.median(tpf.flux[j])) # subtracts bkg
                    matrix[i][j][0] = com[0]+xy[i][0]
                    matrix[i][j][0] = com[1]+xy[i][1]
            
            for i in range(len(fns)):
                centroids = matrix[:,i]
                if i == 0:
                    initGuess = [0.001, 0.1, -0.1]
                else:
                    initGuess = solution.x
                bnds = ((-0.08, 0.08), (-5.0, 5.0), (-5.0, 5.0))
                solution = minimize(model, initGuess, method='L-BFGS-B', bounds=bnds, options={'ftol':5e-11,
                                                                                               'gtol':5e-05})
                sol = solution.x

                with open(self.corrFile, 'a') as tf:
                    if i == 0:
                        theta, delX, delY = sol[0], sol[1], sol[2]
                    else:
                        theta = oldSol[0] - sol[0]
                        delX  = oldSol[1] - sol[1]
                        delY  = oldSol[2] - sol[2]
                    tf.write('{}\n'.format(str(i) + ' ' + str(theta) + ' ' + str(delX) + ' ' + str(delY)))
                    oldSol = sol
            return

        fns = self.sort_by_date()
        mast, header = fits.getdata(self.dir+fns[0], header=True)
        xy, id = pixel_cone(header, 6*np.sqrt(2), [0.0, 5e-5])
        
        # Makes sure sources are in the file
        inds = np.where( (xy[1]>=44.) & (xy[1]<len(mast)-45.) & (xy[0]>=0.) & (xy[0]<len(mast[0])-41.))[0]
        x, y = xy[0][inds], xy[1][inds]

        isolated = find_isolated(x, y)
        x, y = x[isolated], y[isolated]
        calc_shift()
        return
        

    def make_postcard(self):
        """
        Creates 300 x 300 x n postcards, where n is the number of cadences for a given observing run
        Creates a catalog of the associated header with each postcard for use later
        """
        fns = self.sort_by_date()
        fns = [self.dir+i for i in fns]
        mast, mheader = fits.getdata(fns[0], header=True)
        x, y = np.linspace(45, 2093, 8, dtype=int), np.linspace(0, 2048, 8, dtype=int)
        x_cens, y_cens = [], []
        cat = 'postcard_catalog.txt'
        # Creating a table for the headers of each file
        colnames = list(mheader.keys())
        add  = ['POST_FILE', 'POST_SIZE1', 'POST_SIZE2', 'POST_CENX', 'POST_CENY', 'POST_CEN_RA', 'POST_CEN_DEC']
        for a in add:
            colnames.append(a)

        typeList = np.full(len(colnames), 'S30', dtype='S30')
        t = Table(names=colnames, dtype=typeList)

        for i in range(len(x)-1):
            for j in range(len(y)-1):
                fn = 'postcard_{}_{}-{}_{}-{}.fits'.format(self.sector, self.camera, self.chip, i, j)
                x_cen = (x[i]+x[i+1]) / 2.
                y_cen = (y[j]+y[j+1]) / 2.

                radec = WCS(mheader).all_pix2world(x_cen, y_cen, 1)
                s = 300
                tpf   = ktpf.from_fits_images(images=fns, position=(x_cen,y_cen), size=(s,s))
                tpf.to_fits(output_fn=fn)

                # Edits header of FITS files
#                fits.setval(fn, 'CEN_X' , value=np.round(x_cen,5))
#                fits.setval(fn, 'CEN_Y' , value=np.round(y_cen,5))
#                fits.setval(fn, 'CEN_RA', value=float(radec[0]))
#                fits.setval(fn, 'CEN_DEC', value=float(radec[1]))
                
                tempVals = list(mheader.values())
                moreData = [fn, s, s, x_cen, y_cen, float(radec[0]), float(radec[1])]
                for m in moreData:
                    tempVals.append(m)
                t.add_row(vals=tempVals)
        ascii.write(t, output='postcard.cat')
        return


    def find_postcard(self):
        """ 
        Finds what postcard a source is located in
        Returns
        ----------
            postcard filename, header of the postcard
        """
        t = Table.read('postcard.cat', format='ascii.basic')
        in_file=[None]
        # Searches through rows of the table
        for i in range(len(t)):
            data=[]
            # Creates a list of the data in a row
            for j in range(146):
                data.append(t[i][j])
            d = dict(zip(t.colnames[0:146], data))
            hdr = fits.Header(cards=d)
            xy = WCS(hdr).all_world2pix(pos[0], pos[1], 1)
            x_cen, y_cen, l, w = t['POST_CENX'][i], t['POST_CENY'][i], t['POST_SIZE1'][i]/2., t['POST_SIZE2'][i]/2.
            # Checks to see if xy coordinates of source falls within postcard
            if (xy[1] >= x_cen-l) & (xy[1] <= x_cen+l) & (xy[0] >= y_cen-w) & (xy[0] <= y_cen+w):
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
        return t['POST_FILE'][in_file[0]], t[in_file[0]]


    def individual_tpf(self):
        """
        Creates a FITS file for a given source that includes:
            Extension[0] = header
            Extension[1] = (9x9xn) TPF, where n is the number of cadences in an observing run
            Extension[2] = (3 x n) time, raw flux, systematics corrected flux
        """
        
        def init_shift():
#            nonlocal xy
            """ Offsets (x,y) coords of source by pointing model """
            pm = np.loadtxt('pointingModel_{}-{}.txt'.format(self.camera, self.chip), skiprows=1,
                            usecols=(1,2,3))[0]
            pm[0] = np.radians(pm[0])
            x = xy[0]*np.cos(pm[0]) - xy[1]*np.sin(pm[0]) - pm[1]
            y = xy[0]*np.sin(pm[0]) + xy[1]*np.cos(pm[0]) + pm[2]
            return np.array([x,y])     
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
            tpf_com = tpf[0][2:7, 2:7]
            com = ndimage.measurements.center_of_mass(tpf_com.T - np.median(tpf_com))
#            com=ndimage.measurements.center_of_mass(tpf_com - np.median(tpf_com))
            return len(tpf_com)/2-com[0], len(tpf_com)/2-com[1]

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

            # Creates a complete, systematics corrected light curve for each aperture
            for i in range(len(r_list)):
                lc_circ = self.system_corr(matrix[i][0], x, y, jitter=True, roll=True)
                sigma[i][0] = np.std(lc_circ)
                lc_rect = self.system_corr(matrix[i][1], x, y, jitter=True, roll=True)
                sigma[i][1] = np.std(lc_rect)
            best = np.where(sigma==np.min(sigma))
            r_ind, s_ind = best[0][0], best[1][0]
            lc_best = matrix[r_ind][s_ind]
            return r_list[r_ind], s_ind, lc_best, matrix[r_ind][s_ind]

        file_cen = len(tpf[0])/2.
        
        initParams = np.loadtxt(self.corrFile, skiprows=1, usecols=(1,2,3))[0]
        theta, delX, delY = np.loadtxt(self.corrFile, skiprows=1, usecols=(1,2,3), unpack=True)

        startX, startY = centroidOffset(tpf, file_cen)
        startX, startY = file_cen+startX, file_cen+startY

        x, y = [startX], [startY]

        for i in range(1,len(theta)):
#            if i == 0:
#                x.append( startX*np.cos(theta[i]) - startY*np.sin(theta[i]) - delX[i] )
#                y.append( startX*np.sin(theta[i]) + startY*np.cos(theta[i]) - delY[i] )
#            else:
            x.append( x[i-1]*np.cos(theta[i]) - y[i-1]*np.sin(theta[i]) - delX[i] )
            y.append( x[i-1]*np.sin(theta[i]) + y[i-1]*np.cos(theta[i]) - delY[i] )

        radius, shape, lc, uncorr = findLC(x, y)

        return radius, shape, lc, uncorr


    def system_corr(self, lc, x_pos, y_pos, jitter=False, roll=None):
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
            lc_corrected = sff.correct(time, lc, x_pos, y_pos, niters=1,
                                       windows=1, polyorder=5)
            return lc_corrected.flux
        print(lc)
        if jitter==True and roll==True:
            newlc = jitter_corr(lc, x_pos, y_pos)
#            newlc = rotation_corr(newlc, x_pos, y_pos)
        elif jitter==False and roll==True:
            lc = rotation_corr(lc, x_pos, y_pos)
        return newlc



##########################
##########################
##########################
class visualize:
    """
    The main interface for creating figures, movies, and interactive plots
    Allows the user to have a grand ole time playing with their data!
        
    Args:
        tpf: A FITS file that contains stacked cadences for a single source
    """

    def __init__(self, id, dir=None, fn=None, **kwargs):
        """ USER INPUT """
        self.id  = id
        if dir==None:
            self.tpf = fn
        else:
            self.tpf = dir+ fn
                    

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
        tp = ktpf.from_fits(self.tpf)
        lc = tp.to_lightcurve()
#        lc = fits.getdata(self.lcf)
        time, lc = lc.time, lc.flux/np.nanmedian(lc.flux)

        if 'vmax' not in kwargs:
            kwargs['vmax'] = np.max(tp.flux[0])
        if 'vmin' not in kwargs:
            kwargs['vmin'] = np.min(tp.flux[0])

        print(kwargs)
        def animate(i):
            nonlocal line

            ax.imshow(tp.flux[i], origin='lower', **kwargs)
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
            plt.colorbar(plt.imshow(tp.flux[0], **kwargs), ax=ax)

        if output_fn == None:
            output_fn = '{}.mp4'.format(self.id)        
            
        plt.tight_layout()
        ax.set_title('{}'.format(self.id), fontweight='bold')
        return ani


    def click_aperture(self):
        """
        Allows the user to click specific pixels they want to create
        a lightcurve for
        """
        def click_pixels():
            nonlocal tpf
            """ Creates a rectangle over a pixel when that pixel is clicked """
            coords, rectList = [], []

            fig, ax = plt.subplots()
            ax.imshow(tpf.flux[0], origin='lower')
            
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
            ax.imshow(tpf.flux[0], origin='lower')
            
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
            def get_status(val):
                nonlocal good
                if value == 'Yes':
                    good=True
                else:
                    good=False

            butRadio.on_clicked(get_status)
            plt.show()
            return good

        tpf = ktpf.from_fits(self.tpf)
        coords, rectList = click_pixels()
        check = check_pixels()
        if check==True:
            lc = []
            for f in range(len(tpf.flux)):
                cadence = []
                for i in range(len(coords)):
                    cadence.append(tpf.flux[f][coords[i][0], coords[i][1]])
                lc.append(np.sum(cadence))
            lc = np.array(lc) / np.nanmedian(lc)
        else:
            self.click_aperture()
            
        return lc

    def mark_gaia(self):
        """
        Allows the user to mark other Gaia sources within a given TPF
        Click on the x's to reveal the source's ID and Gmag
        Also cross-matches with TIC and identifies sources in there as well
        """
        

