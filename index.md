## Welcome to eleanor

The purpose of eleanor is to go from [TESS](https://heasarc.gsfc.nasa.gov/docs/tess/) Full Frame Images (FFIs) to extracted and systematics-corrected light curves for any given star observed by TESS. In its simplest form, eleanor takes a TIC ID, a Gaia source ID, or (RA, Dec) coordinates of a star observed by TESS and returns, as a single object, a light curve and accompanying target pixel data. There’s plenty of customizability along the way, though: you can examine intermediate data products, change the aperture used for light curve extraction, and much more. The quickstart tutorial goes through these procedures in more detail.


## eleanor Data Products
#### TargetData Objects

The primary data product, that the typical user analyzing a single point source will likely interact with the most, is a `TargetData` object. This is the equivalent of a Kepler Target Pixel File and Light Curve all rolled into one. Saving this object records a FITS file. In the 0th extension, the pixel-level data is stored. Users might find this extension similar to a Kepler “target pixel” file. In the 1st extension is the light curve for eleanor’s estimate of the optimal aperture for the target, both with and without accounting for instrumental systematics. Users might find this extension similar to a Kepler “light curve” file. The results found with other apertures are stored in additional extensions. If a target is observed for multiple sectors, eleanor will produce a target pixel file and light curve for the star for every sector that it is observed. Alternatively, the user can request data for only a specific sector.


#### Postcards

The TESS Full Frame Images are delivered to MAST as .FITS files for a given camera/CCD chip combination, one for every cadence. For each cadence there are then 16 FFIs, and data for a given target for a sector is spread across approximately 1300 FFIs. To reduce the burden of downloading and interacting with 1300 separate files to understand the light curve of one object, eleanor provides “postcards” which contain a smaller region of the detector, stacked for all cadences in a sector. These postcards then have the potential to contain many postage stamps, in order to analyze data from multiple targets simultaneously to better understand systematic effects in the data. When a request is submitted to eleanor to get data for a target, it downloads a postcard rather than every FFI, cuts the requested pixels that a target falls on, and produces a postage stamp and light curve from these data. 


#### NASA GSFC-eleanor-lite FITS files

- Bulleted
- List

1. Numbered
2. List

## Tutorial Video
<iframe width="560" height="315" src="[https://www.youtube.com/embed/dQw4w9WgXcQ](https://www.youtube.com/watch?v=xpvniFrA6V0&t=330s)" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>


**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/afeinstein20/eleanor/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
