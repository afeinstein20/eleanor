#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import sys

import setuptools
from setuptools import setup

sys.path.insert(0, "eleanor")
from version import __version__  


long_description = \
    """
eleanor is a python package to extract target pixel files from
TESS Full Frame Images and produce systematics-corrected light curves
for any star observed by the TESS mission. In its simplest form, eleanor
takes a TIC ID, a Gaia source ID, or (RA, Dec) coordinates of a star
observed by TESS and returns, as a single object, a light curve and
accompanying target pixel data.
Read the documentation at https://adina.feinste.in/eleanor

Changes to v0.2.10 (2019-11-07):

* Updated to include Sector 16
* Handling edge cases with TessCut
* Other bug fixes
"""



setup(
    name='eleanor',
    version=__version__,
    license='MIT',
    author='Adina D. Feinstein',
    author_email='adina.d.feinstein@gmail.com',
    packages=[
        'eleanor',
        ],
    include_package_data=True,
    url='http://github.com/afeinstein20/eleanor',
    description='Source Extraction for TESS Full Frame Images',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={'': ['README.md', 'LICENSE']},
    install_requires=[
        'mplcursors', 'photutils', 'tqdm', 'lightkurve>=1.1.0', 'astropy',
        'astroquery', 'bokeh', 'muchbettermoments', 'fitsio', 'pandas',
        'setuptools>=41.0.0', 
        'tensorflow<=1.14.0', 'vaneska', 'beautifulsoup4>=4.6.0', 'tess-point'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.0',
        ],
    )
