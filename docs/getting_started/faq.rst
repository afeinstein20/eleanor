.. _faq:

Frequently Asked Questions
==========================

* How does eleanor work?
	Under the hood, when you query an object, eleanor performs the following steps:
    
        #. Determines in which sectors this target was observed.
        #. Locates the object on TESS's many cameras/chips.
        #. Downloads a time series of "postcards" containing TESS data for the object and its immediate surroundings.
        #. Creates and stores a target pixel file (TPF) of the object.
        #. Traces centroid shifts for the object across the time series.
        #. Chooses an optimal pixel aperture for photometry.
        #. Creates a light curve using the chosen aperture and centroid trace.
        #. Performs basic systematics corrections on the light curve and stores it.
		
* What final data products do I get from eleanor?
	By default, eleanor delivers the systematics-corrected light curve of your source and the source's TPF. If you are happy with what you see there, you're done and ready to science! 
	
* How can I customize eleanor for my source?
	If you want to dig a little bit deeper, eleanor delivers several other data products to help you do that. The TPF carries with it a trace of the source's centroid through time and a record of the aperture used, as well as light curves developed using other apertures, which may be more optimal for your specific science application. This makes it easier for you to check for possible issues and reproduce the analysis. eleanor also has tools to define your own aperture, visualize the light curve, and more. See the :ref:`quickstart tutorial <quickstart tutorial>` for examples.
	
* What is a target pixel file (TPF)?
	Target pixel files are essentially a stack of images of the source, containing one image per observed FFI. They differ from postcards because they're only a few pixels on a side. For more information, see `lightkurve <https://lightkurve.keplerscience.org/tutorials/1.02-target-pixel-files.html>`_. TPF objects in eleanor carry additional attributes: the centroid trace and the aperture used to create the light curve. Taken together, these provide all the information you need to reproduce the light curve.
	
* What are eleanor's intermediate data products and how can I access them?
	eleanor is designed to save you, the user, from having to download the entire TESS FFIs to get at a small region of sky. To do this, eleanor produces "postcard" files which are more manageable subsets of each detector chip. These postcards can be downloaded and accessed through the eleanor :ref:`API <api>`. Postcards also carry with them an associated pointing model.
	
* What is the pointing model?
	The pointing model consists of a few simple parameters that describe how to go from TESS pixels to an accurate world coordinate system. eleanor optimizes this pointing model using bright stars across the detector. Most users will never need to touch the pointing model, though - its information gets encoded in the centroid trace for any given source.

* What kinds of systematic noise in the light curves can eleanor correct?
	By default, eleanor corrects the light curve by regressing out jitter correlated with the position of the telescope in time. Future versions of eleanor will include more advanced detrending that makes use of all sources on a given postcard or chip.

* My light curve has weird features. What gives?
	Here are a few things you can easily check on: are there changes in the object centroid coincident with the features? Does the aperture choice make sense? Are there blended background sources? Do you see the same features in all apertures, or in other nearby stars at the same time?
	
	If these checks don't turn up anything and you suspect a problem with the code, feel free to open a `Git Issue`_.

* How do I cite eleanor in a published work?
    If you find eleanor light curves useful for your work, please cite Feinstein et al. 2019, PASP in press. A preprint can be found at arXiv:`1903.09152`_.


.. _Git Issue: http://github.com/afeinstein20/eleanor/issues
.. _1903.09152: https://arxiv.org/abs/1903.09152
