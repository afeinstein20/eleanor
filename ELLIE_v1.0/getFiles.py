import os, sys
from bs4 import BeautifulSoup
import requests
import numpy as np

# --------------------------
# Finds all calibrated files on webpage
# -------------------------- 
def getAllFiles(year, daynum, camera, chip):
    url = 'https://archive.stsci.edu/missions/tess/ete-6/ffi/{0}/{1}/{2}-{3}/'.format(year, daynum, camera, chip)

    soup = BeautifulSoup(requests.get(url).text, "lxml")
    calFiles = []
    for link in soup.find_all('a'):
        filename = link.get('href')
        if filename[-7::] == 'ic.fits':
            calFiles.append(url + filename)
    return calFiles
    
# --------------------------  
# Makes sure directory exists
# -------------------------- 
def dirExists(directory):
    if os.path.isdir(directory) == False:
        os.makedir(directory)
    return

# -------------------------- 
# Downloads all calibrated files
# -------------------------- 
def scrapeFiles(directory, filenames):
    dirExists(directory)
    for i in filenames:
        os.system('curl -O -L {}'.format(i))

# -------------------------- 
#      Gets user input
# -------------------------- 
def usrInput():
    year   = str(sys.argv[1])
    daynum = str(sys.argv[2])
    camera = str(sys.argv[3])
    chip   = str(sys.argv[4])
    return year, daynum, camera, chip

# --------------------------   
# Used to call downloading routine
# --------------------------   
def download():
    year, daynum, camera, chip = usrInput()
    directory = './calFits_{}_{}-{}-{}/'.format(year, daynum, camera, chip)

#    for camera in np.arange(1,5,1):
#        for chip in np.arange(1,5,1):
    camera, chip = 3, 3
    filenames = getAllFiles(year, daynum, camera, chip)
    scrapeFiles(directory, filenames)
    filenames = os.listdir(directory)

# -------------------------- 
# Gets files in calibrated files directory
# without re-downloading everything
# -------------------------- 
def passInfo():
    year, daynum, camera, chip = usrInput()
    directory = './calFits_{}_{}-{}-{}/'.format(year, daynum, camera, chip)
    filenames = os.listdir(directory)
    return year, daynum, filenames, directory, camera, chip

