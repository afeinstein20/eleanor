import os, sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import requests
from bs4 import BeautifulSoup

class server_tpfs:
    """
    This class will be used on a cluster for mass download of the TESS
        FFIs for bulk creation of TESS Input Catalog sources TPFs
    These TPFs and light curves will be hosted on the MAST servers and
        will be available for public download
    """
    def __init__(self, sector=None, camera=None, chip=None, url=None, dayMin=None, dayMax=None):
        if sector == None:
            self.sector=-1.
        else:
            self.sector=sector
        if camera == None:
            self.camera=0
        else:
            self.camera=camera
        if chip == None:
            self.chip=0
        else:
            self.chip=chip
        if url == None:
            self.url=''
        else:
            self.url=url
        if dayMin==None:
            self.dayMin=129
        else:
            self.dayMin=dayMin
        if dayMax==None:
            self.dayMax=156
        else:
            self.dayMax=dayMax


    
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


     def find_all_fits(self):
         
