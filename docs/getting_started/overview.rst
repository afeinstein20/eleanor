.. _overview:

Overview
========

Summary of ELLIE Functionality
------------------------------

The purpose of ELLIE is to go from TESS Full Frame Images to extracted and systematics-corrected light curves for any given star observed by TESS. 
In its simplest form, ELLIE takes a TIC ID, a Gaia source ID, or (RA, Dec) coordinates of a star observed by TESS and returns a light curve and accompanying target pixel file. 
There's plenty of customizability along the way, though: you can examine intermediate data products, change the aperture used for light curve extraction, and much more. 
The :ref:`tutorial <tutorial>` goes through these procedures in more detail.


FAQ
---

* How does ELLIE work?
	Under the hood, when you query an object, ELLIE performs the following steps:
		* Locates the object on TESS's many cameras/chips.
		* Downloads a time series of "postcards" containing TESS data for the object and its immediate surroundings.
		* Creates and stores a target pixel file (TPF) of the object.
		* Traces centroid shifts for the object across the time series.
		* Chooses an optimal pixel aperture for photometry.
		* Creates a light curve using the chosen aperture and centroid trace.
		* Performs basic systematics corrections on the light curve and stores it.
		
* What final data products do I get from ELLIE?
	By default, ELLIE delivers the systematics-corrected light curve of your source and the source's TPF. If you are happy with what you see there, you're done and ready to science! 
	
* How can I customize ELLIE for my source?
	If you want to dig a little bit deeper, ELLIE delivers several other data products to help you do that. The TPF carries with it a trace of the source's centroid through time and a record of the aperture used. This makes it easier for you to check for possible issues and reproduce the analysis. ELLIE even has tools to help you change the aperture used and more. See the :ref:`tutorial <tutorial>` for examples.
	
* What is a target pixel file (TPF)?
	Target pixel files are essentially a stack of images of the source, one image per FFI. They differ from postcards because they're only a few pixels on a side. For more information, see `lightkurve <https://lightkurve.keplerscience.org/tutorials/1.02-target-pixel-files.html>`_. TPF objects in ELLIE carry additional attributes: the centroid trace and the aperture. Taken together, these provide all the information you need to reproduce the light curve.
	
* What are ELLIE's intermediate data products and how can I access them?
	ELLIE is designed to save you, the user, from having to download the entire TESS FFIs to get at a small region of sky. To do this, ELLIE produces "postcard" files which are more manageable subsets of each detector chip. These postcards can be downloaded and accessed through the ELLIE :ref:`API <api>`. Postcards also carry with them an associated pointing model.
	
* What is the pointing model?
	The pointing model consists of a few simple parameters that describe how to go from TESS pixels to an accurate world coordinate system. ELLIE optimizes this pointing model using bright stars across the postcard. Most users will never need to touch the pointing model, though - its information gets encoded in the centroid trace for any given source.

* What kinds of systematic noise in the light curves can ELLIE correct?
	By default, ELLIE corrects the light curve by 1) regressing out jitter using a quadratic fit to the centroid trace and 2) correcting for spacecraft roll with the self-flat-fielding (SFF) method as implemented in `lightkurve <https://lightkurve.keplerscience.org/api/lightkurve.correctors.SFFCorrector.html>`_. Future versions of ELLIE may include more advanced detrending that makes use of all sources on a given postcard or chip.

* My light curve has weird features. What gives?
	Here are a few things you can easily check on: are there changes in the object centroid coincident with the features? Does the aperture choice make sense? Are there blended background sources?
	
	If these checks don't turn up anything and you suspect a problem with the code, feel free to open a `Git Issue`_.

* How do I cite ELLIE in a published work?


.. _Git Issue: http://github.com/afeinstein20/ELLIE/issues