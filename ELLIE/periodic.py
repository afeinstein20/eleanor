from bokeh.models import HoverTool
from bokeh.plotting import figure, show, output_file
from bokeh.sampledata.periodic_table import elements
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, BasicTicker
from bokeh.transform import dodge, factor_cmap
import numpy as np
from astropy.io import fits
from ellie import data_products as dp
from ellie import find_sources
from astropy.wcs import WCS

def find_gaia_sources(header, hdr):
    pos = [header['SOURCE_RA'], header['SOURCE_DEC']]
    l   = find_sources(pos=pos)
    sources = l.cone_search(r=0.5, service='Mast.Catalogs.GaiaDR2.Cone')
    xy = WCS(hdr).all_world2pix(sources['ra'], sources['dec'], 1)
    return xy, sources['source_id'], sources['phot_g_mean_mag']


def in_tpf(xy, gaiaXY, gaiaID, gaiaMAG):
    """ Pushes the gaia sources to the appropriate place in the TPF """
    gaiaX, gaiaY = gaiaXY[0]-xy[0]+4.5, gaiaXY[1]-xy[1]+6.5
    inds = np.where( (gaiaX >= -0.5) & (gaiaX <= 8.5) &
                     (gaiaY >= -0.5) & (gaiaY <= 8.5) & (gaiaMAG <= 16.5))
    return [gaiaX[inds], gaiaY[inds]], gaiaID[inds], gaiaMAG[inds]


def pointingCorr(xy, header, pm):
    """ Corrects (x,y) coordinates based on pointing model """
    shift = pm[0]
    x = xy[0]*np.cos(shift[0]) - xy[1]*np.sin(shift[0]) - shift[1]
    y = xy[0]*np.sin(shift[0]) + xy[1]*np.cos(shift[0]) - shift[2]
    return np.array([x,y])


output_file("gaia_hover.html")

tic = 198593129
hdu = fits.open('hlsp_ellie_tess_ffi_{}_v1_lc.fits'.format(tic))
tpf = hdu[0].data[0]
header = hdu[0].header
center = [header['SOURCE_X'], header['SOURCE_Y']]

# Get online things
e = dp(tic=tic)
hdr = e.get_header(postcard=header['POSTCARD'])
pm  = e.get_pointing(header=header)

# Find & correct gaia sources
gaia_xy, gaia_id, gaia_gmag = find_gaia_sources(header, hdr)
gaia_xy = pointingCorr(gaia_xy, header, pm)
gaia_xy, gaia_id, gaia_gmag = in_tpf(center, gaia_xy, gaia_id, gaia_gmag)

# Create labels
ticLabel, tmagLabel = np.zeros(len(gaia_id.data)), np.zeros(len(gaia_id.data))

c = find_sources()
crossTable = c.crossmatch_multi_to_tic(list=gaia_id.data)

for i in range(len(gaia_id.data)):
    row = crossTable[i]
    if row['separation'] <= 1.0 and row['Gmag'] <= 16.5:
        ticLabel[i]  = row['TIC_ID']
        tmagLabel[i] = row['Tmag']


hover = HoverTool()
hover.tooltips = [
    ("TIC ID", "test"),
    ("T_mag", "test"),
    ("GAIA ID", "test"),
    ("G_mag", "test")
    ]

p = figure(x_range=(0,9), y_range=(0,9), 
           plot_width=450, plot_height=450)

color_mapper = LinearColorMapper(palette='Viridis256', low=np.min(tpf), high=np.max(tpf))
tpf_img = p.image(image=[tpf], x=0, y=0, dw=9, dh=9, color_mapper=color_mapper)
color_bar = ColorBar(color_mapper=color_mapper, location=(0,0), border_line_color=None,
                     ticker=BasicTicker())
p.add_layout(color_bar, 'right')

source=ColumnDataSource(data=dict(x=gaia_xy[0], y=gaia_xy[1],
                                  tic=ticLabel, tmag=tmagLabel,
                                  gaia=gaia_id, gmag=gaia_gmag))


s = p.circle('x', 'y', size=8, source=source, line_color=None,
             selection_color='red', nonselection_fill_alpha=0.0, nonselection_line_alpha=0.0,
             nonselection_line_color=None, fill_color='black', hover_alpha=0.9, hover_line_color='white')
p.add_tools(HoverTool( tooltips=[("TIC Source", "@tic"), ("Tmag", "@tmag"), 
                                 ("Gaia Source", "@gaia"), ("Gmag", "@gmag")],
                       renderers=[s], mode='mouse', point_policy="snap_to_data"))


show(p)

