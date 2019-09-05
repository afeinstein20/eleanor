<p align="center">
  <img width = "600" src="./figures/eleanor.gif"/>
</p>
<p align="center">  
  <a href="https://travis-ci.org/afeinstein20/eleanor/"><img src="https://img.shields.io/travis/afeinstein20/eleanor/master.svg?colorB=F9E84F"/></a>
  <a href="https://afeinstein20.github.io/eleanor/"><img src="https://img.shields.io/badge/read-the_docs-4D827F.svg?style=flat"/></a>
  <a href="https://ui.adsabs.harvard.edu/abs/2019PASP..131i4502F/abstract"><img src="https://img.shields.io/badge/read-the_paper-4D827F.svg?style=flat"/></a>
  <a href="https://doi.org/10.5281/zenodo.2597620"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.2597620.svg?colorB=3C0650" alt="DOI"></a>
</p>

<div align="justify">
eleanor is a python package to extract target pixel files from TESS Full Frame Images and produce systematics-corrected light curves for any star observed by the TESS mission. In its simplest form, eleanor  takes a TIC ID, a Gaia source ID, or (RA, Dec) coordinates of a star  observed by TESS and returns, as a single object, a light curve and  accompanying target pixel data.
</p>
To install eleanor with pip:

        pip install eleanor
        

Alternatively you can install the current development version of eleanor:

        git clone https://github.com/afeinstein20/eleanor
        cd eleanor
        python setup.py install

For more information on how to install and use eleanor, please refer to the <a href="http://adina.feinste.in/eleanor/">documentation</a>.

