.. _overview:

About eleanor
=============

Summary of eleanor Functionality
--------------------------------

The purpose of eleanor is to go from TESS Full Frame Images (FFIs) to extracted and systematics-corrected light curves for any given star observed by TESS. 
In its simplest form, eleanor takes a TIC ID, a Gaia source ID, or (RA, Dec) coordinates of a star observed by TESS and returns, as a single object, a light curve and accompanying target pixel data. 
There's plenty of customizability along the way, though: you can examine intermediate data products, change the aperture used for light curve extraction, and much more. 
The :ref:`quickstart tutorial <quickstart tutorial>` goes through these procedures in more detail.

Summary of Data Products
------------------------

* Postcards
        The TESS Full Frame Images are delivered to MAST as .FITS files for a given camera/CCD chip combination, one for every cadence. 
        For each cadence there are then 16 FFIs, and data for a given target for a sector is spread across approximately 1300 FFIs. 
        To reduce the burden of downloading and interacting with 1300 separate files to understand the light curve of one object, eleanor provides
        "postcards" which contain a smaller region of the detector, stacked for all cadences in a sector. 
        These postcards then have the potential to contain many postage stamps, in order to analyze data from multiple targets simultaneously to better
        understand systematic effects in the data.
        When a request is submitted to eleanor to get data for a target, it downloads a postcard rather than every FFI, cuts the requested pixels
        that a target falls on, and produces a postage stamp and light curve from these data.
        An example postcard is shown below.


* TargetData objects
    	The primary data product, that the typical user analyzing a single point source will likely interact with the most, is a `TargetData` object. 
        This is the equivalent of a Kepler Target Pixel File and Light Curve all rolled into one. Saving this object records a FITS file. 
        In the 0th extension, the pixel-level data is stored. Users might find this extension similar to a Kepler "target pixel" file. 
        In the 1st extension is the light curve for eleanor's estimate of the optimal aperture for the target, both with and without accounting for instrumental systematics. 
        Users might find this extension similar to a Kepler "light curve" file.
        The results found with other apertures are stored in additional extensions.
        If a target is observed for multiple sectors, eleanor will produce a  target pixel file and light curve for the star for every sector that it is observed. 
        Alternatively, the user can request data for only a specific sector.



.. _Git Issue: http://github.com/afeinstein20/eleanor/issues
