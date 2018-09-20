# Unit test to make sure WCS is working
# Takes a pre-made postcard, blanks out all the flux, plops in a star
# We want to take the WCS and find that star again to make sure it's identifying 
#    the correct star
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import urllib
from astropy.table import Table, Row, Column
from astropy.wcs import WCS
from ellie import data_products as dp

def get_header(postcard=None):
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
        return hdr, row['POST_CEN_RA'], row['POST_CEN_DEC']



post = 'postcard_1_3-3_0-1.fits'
hdu = fits.open('./.ellie/postcards/{}'.format(post))
hdr, cenRA, cenDEC = get_header(postcard=post)
ra, dec = 274.269351, 48.709335
star_xy = WCS(hdr).all_world2pix(ra, dec, 1)

star_x, star_y = int(np.round(star_xy[0]-cenDEC)), int(np.round(star_xy[1]-cenRA))
print(star_x, star_y)
tpf = np.zeros(np.shape(hdu[0].data))
for cadence in tpf:
    cadence[star_x][star_y] = 1

#hdu1 = fits.PrimaryHDU(header=hdr)
#hdu2 = fits.ImageHDU()
#hdu1.data = tpf
#hdu2.data = hdu[1].data
#new_hdu = fits.HDUList([hdu1, hdu2])
#new_hdu.writeto('postcard_1_3-3_0-1.fits')

#a = dp(tic=21104215)
a = dp(tic=21104252)
a.individual_tpf()
