#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import sys

import setuptools
from setuptools import setup

sys.path.insert(0, "eleanor")
from version import __version__


# define the dependencies
install_reqs = [
    "nbsphinx"
    "ipykernel"
    "tqdm"
    "lightkurve"
    "astropy>=3.2.3"
    "astroquery"
    "pandas"
    "tess-point"
    "astroquery"
    "setuptools>=41.0.0"
    "beautifulsoup4>=4.6.0"
    "jax"
    # photutils will change depending on the python version
    "photutils=1.11.0; python_version < 3.10"
    "photutils=2.0.2; python_verson == 3.10"
    "photutils=3.0.0; python_version >= 3.12"
]

long_description = \
    """
eleanor is a python package to extract target pixel files from
TESS Full Frame Images and produce systematics-corrected light curves
for any star observed by the TESS mission. In its simplest form, eleanor
takes a TIC ID, a Gaia source ID, or (RA, Dec) coordinates of a star
observed by TESS and returns, as a single object, a light curve and
accompanying target pixel data.
Read the documentation at https://adina.feinste.in/eleanor

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
    install_requires=install_reqs,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.0',
        ],
    )
