import sys, os
import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table
import fitsio
import os, glob, tqdm, requests
from bs4 import BeautifulSoup

#FFIURL = 'https://archipelago.uchicago.edu/tess_postcards/eleanor_files/ffis/'

def find_ffis(directory):
    ffis = []
    files = os.listdir(directory)
    files = [directory+i for i in files if 'tess' and '.fits' in i]
    return np.array(files)


def get_headers(cards, exists=False, t=None):
    for i in tqdm.tqdm(range(len(cards)), total=len(cards)):
        try:
            hdr = fits.open(cards[i])[1].header
            key = 'TMOFST{0}{1}'.format(hdr['CAMERA'], hdr['CCD'])
            
            sector = os.path.split(cards[i])[-1].split("-")[1]

            # Initiate table using first postcard                                                           
            if i == 0 and exists == False:
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
                
            elif i == 0 and exists == True:
                names = t.colnames
                dtype = ['S60' for _ in range(len(names))]
                counts = np.arange(0,182,1)
                hdrKeys = [k for k in hdr.keys() if not k.startswith('TMOFST')]

            row = np.array([hdr.get(k, 0) for k in hdrKeys])
            row = row[counts]
            row = np.append(row, cards[i])
            row = np.append(row, sector)
            row = np.append(row, hdr[key])

            if exists == True:
                sec_ind = [i for i in np.arange(0,len(names),1) if 'SECTOR' in names[i]][0]
                cam_ind = [i for i in np.arange(0,len(names),1) if 'CAMERA' in names[i]][0]
                ccd_ind = [i for i in np.arange(0,len(names),1) if 'CCD'    in names[i]][0]
                file_check = [row[sec_ind], int(row[cam_ind]), int(row[ccd_ind])]
                
                checker = 0
                for j in range(len(t)):
                    check = [t['SECTOR'][j], t['CAMERA'][j], t['CCD'][j]]
                    if check==file_check:
                        checker += 1
                if checker == 0:
                    t.add_row(row)
                        
            elif exists == False:
                t.add_row(row)

        except:
            print(key, cards[i])
            raise
    return t


if len(sys.argv) > 1:
    dirname = sys.argv[1]
    ffidir  = sys.argv[2]
else:
    dirname = '.'
    ffidir  = './ffis/'

fn_output = 'ffi.guide'
fn_path   = os.path.join(dirname, fn_output)

ffis = find_ffis(ffidir)
exists = os.path.isfile(os.path.join(dirname, fn_output))

if exists:
    t = Table.read(fn_path, format='ascii')
    table = get_headers(ffis, exists, t)
else:
    table = get_headers(ffis)

ascii.write(table, os.path.join(dirname, 'ffi.guide'), overwrite=True)

t = Table.read(fn_path, format='ascii')
print(len(t['SECTOR']))
