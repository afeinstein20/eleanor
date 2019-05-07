import sys
import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table
import fitsio
import os, glob, tqdm, requests
from bs4 import BeautifulSoup

FFIURL = 'https://archipelago.uchicago.edu/tess_postcards/eleanor_files/ffis/'

def find_ffis():
    ffis = []
    paths = BeautifulSoup(requests.get(FFIURL).text, "lxml").find_all('a')
    for p in paths:
        fn = p.get('href')
        if 'tess' and '.fits' in fn:
            ffis.append(os.path.join(FFIURL, fn))
    return np.array(ffis)


def get_headers(cards):
    for i in tqdm.tqdm(range(len(cards)), total=len(cards)):
        try:
            hdr = fits.open(cards[i])[1].header
            key = 'TMOFST{0}{1}'.format(hdr['CAMERA'], hdr['CCD'])

            sector = os.path.split(cards[i])[-1].split("-")[1]

            # Initiate table using first postcard                                                           
            if i == 0:
                names, counts = [], []
                hdrKeys = [k for k in hdr.keys() if not k.startswith('TMOFST')]
                dtype = []
                for k in range(len(hdrKeys)):
                    if hdrKeys[k] not in names:
                        names.append(hdrKeys[k])
                        counts.append(k)
                names.append('FFINAME')
                names.append('SECTOR')
                names.append('TMOFST')

                counts = np.array(counts)

                dtype = ['S60' for _ in range(len(names))]
                t = Table(names=names, dtype=dtype)

            row = np.array([hdr.get(k, np.nan) for k in hdrKeys])
            row = row[counts]
            row = np.append(row, cards[i])
            row = np.append(row, sector)
            row = np.append(row, hdr[key])
            t.add_row(row)
        except:
            print(key, cards[i])
            raise
    return t


if len(sys.argv) > 1:
    dirname = sys.argv[1]
else:
    dirname = "."

ffis = find_ffis()
table = get_headers(ffis)

ascii.write(table, os.path.join(dirname, 'ffi.guide'), overwrite=True)
