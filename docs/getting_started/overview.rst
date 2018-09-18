.. _overview:

Overview
========

Summary of ELLIE Functionality
------------------------------

The purpose of ELLIE is to go from TESS Full Frame Images to extracted and systematics-corrected light curves for any given star observed by TESS. 
In its simplest form, ELLIE takes a TIC ID (or a Gaia source ID) of a star observed by TESS and returns a light curve. 
There's plenty of customizability along the way, though: you can examine intermediate data products, change the aperture used for light curve extraction, and much more. 
The :ref:`tutorial <tutorial>` goes through these procedures in more detail.


FAQ
---

* How does ELLIE work?
	magic
	
* What are ELLIE's intermediate data products and how can I access them?
	To save us from having to read in the entire FFI, ELLIE produces "postcard" files which are more manageable subsets of each detector chip. These postcards can be downloaded and accessed through the ELLIE :ref:`API <api>`.
	
* What is the pointing model?

* What kinds of systematic noise in the light curves can ELLIE correct?

* My light curve looks funky. What gives?

* How do I cite ELLIE in a published work?


