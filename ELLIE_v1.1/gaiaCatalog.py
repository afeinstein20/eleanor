import os, sys
import requests
from bs4 import BeautifulSoup
import numpy as np

path    = 'http://cdn.gea.esac.esa.int/Gaia/gdr2/gaia_source/csv/'
catalog = 'gaiaSources.cat'
soup    = BeautifulSoup(requests.get(path).text, "lxml").find_all('a')

for fn in soup:
    fn = fn.get('href')
    if fn[-7::] == '.csv.gz':
        os.system('curl -O -L {}'.format(path+fn))
        data = np.loadtxt(fn, dtype = str, delimiter = ',')

        inds = np.where(data[:,50] <= '16.5')[0]
        id  = data[:,2]
        ra  = data[:,5]
        dec = data[:,7]
        mag = data[:,50]
        for i in inds:
            row = [id[i], ra[i], dec[i], mag[i]]
            with open(catalog, 'a') as tf:
                tf.write('{}\n'.format(' '.join(e for e in row)))
        os.remove(fn)
