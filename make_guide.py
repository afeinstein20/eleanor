import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table
import fitsio
import os


def postcard_names(loc):
    files = os.listdir(loc)
    cards = []
    for f in files:
        if 'postcard' in f:
            cards.append(f)
    return np.array(cards)


def get_headers(cards):
    colnames = np.array(['SECTOR', 'CAMERA', 'CCD', 'POSTPIX1', 'POSTPIX2',
                         'POST_HEIGHT', 'POST_WIDTH', 'CEN_X', 'CEN_Y', 'POST_NAME'])

    t = Table(names=colnames, dtype=('f4', 'f4', 'f4', 'f4', 
                                     'f4', 'f4', 'f4', 'f4', 
                                     'f4', 'S60'))
    for c in cards:
        data, header = fitsio.read(c, 1, header=True)
        row = []
        for n in colnames[0:len(colnames)-1]:
            row.append(header[n])
        row.append(c)
        t.add_row(row)
    return t


postcards = postcard_names('.')
table = get_headers(postcards)

ascii.write(table, 'postcard.guide')
