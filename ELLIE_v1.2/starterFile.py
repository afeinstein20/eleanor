import os, sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import requests
from bs4 import BeautifulSoup

# -------------------------- 
# Finds all calibrated files from URL
#  Returns: All files for each chip
# -------------------------- 
def findAllFITS(year, camera, chip, url, dayMin, dayMax):
    calFiles = []
    days = np.arange(dayMin, dayMax+1, 1)
    for d in days:
        path = url + '/' + year + '/' + str(d) + '/' + camera + '-' + chip + '/'
        for fn in BeautifulSoup(requests.get(path).text, "lxml").find_all('a'):
            if fn.get('href')[-7::] == 'ic.fits':
                calFiles.append(path + fn.get('href'))
        print(calFiles)
    return calFiles
        
# -------------------------- 
# Makes new directory & downloads
#  files into new directory
# -------------------------- 
def downloadFiles(dir, fns):
    if os.path.isdir(dir) == False:
        os.mkdir(dir)
    for i in fns:
        os.system('cd {} && curl -O -L {}'.format(dir, i))
    return

# -------------------------- 
# Gets starting information from user
# -------------------------- 
def userInput():
    year   = str(sys.argv[1])
    dayMin = int(sys.argv[2])
    dayMax = int(sys.argv[3])
    camera = str(sys.argv[4])
    chip   = str(sys.argv[5])
    url    = str(sys.argv[6])
    dir    = './calFits_{}_{}-{}/'.format(year, camera, chip)
    return year, dayMin, dayMax, camera, chip, url, dir

# -------------------------- 
# Obtains necessary info for 
#   starting FFI analysis
# -------------------------- 
def passInfo():
    year, dayMin, dayMax, camera, chip, url, dir = userInput()
    if os.path.isdir(dir) == False:
        filenames = findAllFITS(year, camera, chip, url, dayMin, dayMax)
        downloadFiles(dir, filenames)
    filenames = os.listdir(dir)
    return year, camera, chip, np.array(filenames), dir

passInfo()
