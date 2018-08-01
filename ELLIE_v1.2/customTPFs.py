import os, sys, re, json, time
import requests
from bs4 import BeautifulSoup
import numpy as np
import matplotlib

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

from photutils import CircularAperture, RectangularAperture, aperture_photometry
from lightkurve import KeplerTargetPixelFile as ktpf
from scipy import ndimage
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

    def __init__(self, tic=None, gaia=None, pos=None, multiFile=None):
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

    
    def pointing_model(self):
        
        def sortByDate(fns, dir):
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


        def findIsolated(x, y):
            """
            Find isolated sources in the image where an isolated source is 
            >= 5.5 pixels away from any other source
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
                if dist >= 5.5:
                    isolated.append(i)
            return isolated
