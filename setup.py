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

Changes to v2.0.1 (2020-12-08):
* Minor bug fixes

Changes to v2.0.0rc1 (2020-10-05):
* Changing skip parameter in correction to be in unit of time rather than cadences to handle 10-minute FFIs
* Fixed bug where very faint stars had suboptimal corrections
* Changed `crowded_field` to `aperture_mode` and added built-in mechanics for bright stars
* Updated docs to include citation to 2019 paper 
* Fixed bug in eleanor.Visualize() layout of aperture
* Changes to handle format of 10-minute FFIs in eleanor.Update()
* Significant speedups, especially when local postcards already exist and run with local==True
* Other bug fixes

Changes to v1.0.5 (2020-05-21):
* Fixed bug where some apertures were made twice and others not at all
* Fixed bug where mass centroids were off by a constant offset at all cadences
* Fixed bug that happens when data are all zeros at the start of a sector

Changes to v1.0.4 (2020-03-27):
* Pass in an array of regressors to use in calculating the corrected flux
* Extreme short-term flux variability like eclipses ignored in corrections, which should improve detrending of these objects
* Fixed bug where metadata could not be properly updated on linux clusters
* Improvements to Gaia overlay of fields
* Ability to create light curves offline
* Ability to pass through a SIMBAD-resolvable name rather than TIC/Gaia ID or coordinates
* Other minor bug fixes

Changes to v1.0.1 (2019-12-19):
* Ability to use local postcards
* Addition of eleanor.Update() for automatic sector updates
* Significant speedups when TIC, Coords, and a Gaia ID are all provided
* Other bug fixes

Changes to v1.0.0 (2020-01-14):
* Pass in the name of the source as a string
* Other bugfixes
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
        'photutils>=0.7', 'tqdm', 'lightkurve>=1.9.0', 'astropy>=3.2.3',
        'astroquery', 'pandas',
        'setuptools>=41.0.0', 'torch', 'tensorflow<2.0.0',
        'beautifulsoup4>=4.6.0', 'tess-point>=0.3.6'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.0',
        ],
    )
