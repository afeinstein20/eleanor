from setuptools import setup
import ellie


setup(
    name='ellie',
    version='1.0',
    license='MIT',
    long_description=open('README.rst').read(),
    author='Adina Feinstein',
    author_email='afeinstein@uchicago.edu',
    packages=[
        'ellie',
        ],
    include_package_data=True,
    url='http://github.com/afeinstein20/ELLIE',
    description='Source Extraction for TESS Full Frame Images',
    package_data={'':['README.rst', 'LICENSE']},
    install_requires=[
        'mplcursors', 'photutils', 'tqdm', 'lightkurve', 'astropy',
        ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.0',
        ],
    )
