from bokeh.layouts import layout, row, column, widgetbox
from bokeh.models import (ColumnDataSource, HoverTool, BasicTicker,
                          Slider, Button, Label, LinearColorMapper, Span,
                          ColorBar)
from bokeh.io import curdoc
from bokeh.palettes import Viridis256
from bokeh.plotting import figure, show
import numpy as np
from astropy.io import fits


def animate_update():
    cadence = int(slider.value) + 1
    if cadence > cadences[-1]:
        cadence = cadences[0]
    slider.value = int(cadence)
    vert.location = int(cadence)

def slider_update(attr, old, new):
    cadence = int(slider.value)
    p.image(image=[tpf[cadence]], x=-0.5, y=-0.5, dw=9, dh=9, color_mapper=color_mapper)
    vert.location = cadence


fn = 'hlsp_ellie_tess_ffi_420888018_v1_lc.fits'
id = 'TIC 420888018'
hdu = fits.open(fn)
tpf = hdu[0].data
cadences = hdu[1].data[0]
hdr = hdu[0].header


# Creates the plot for the TPF movie
xyrange=(-0.5,8.5)
p = figure(x_range=xyrange, y_range=xyrange, title=id,
           plot_height=250, plot_width=300)
p.xaxis.axis_label = 'Pixel Column Number'
p.yaxis.axis_label = 'Pixel Row Number'
color_mapper = LinearColorMapper(palette='Viridis256', low=np.min(tpf), high=np.max(tpf))
p.image(image=[tpf[int(cadences[0])]], x=-0.5, y=-0.5, dw=9, dh=9, color_mapper=color_mapper)
color_bar = ColorBar(color_mapper=color_mapper, location=(0,0), border_line_color=None,
                     ticker=BasicTicker(), title='Intensity', title_text_align='left')
p.add_layout(color_bar, 'right')

# Creates the plot for the light curve movie
l = figure(plot_height=250, plot_width=800)
l.xaxis.axis_label = 'Time'
l.yaxis.axis_label = 'Normalized Flux'
source = ColumnDataSource(data=dict(time=hdu[1].data[0], flux=hdu[1].data[1]))
l.circle('time', 'flux', source=source, size=2, alpha=0.8, color='royalblue')

vert = Span(location=0, dimension='height', line_color='red',
            line_width=2, line_alpha=0.4)
l.add_layout(vert)

# Creates the slider widget
slider = Slider(start=cadences[0], end=cadences[-1], 
                value=cadences[0], step=1, title='TPF Cadence Slice',
                width=300)
slider.on_change('value', slider_update)


callback_id = None
def animate():
    global callback_id
    if button.label == '► Play':
        button.label = '❚❚ Pause'
        callback_id = curdoc().add_periodic_callback(animate_update, 10)
    else:
        button.label = '► Play'
        curdoc().remove_periodic_callback(callback_id)
        
# Creates the button widget
button = Button(label='► Play', width=60)
button.on_click(animate)

layout = layout([
        [l, p],
        [slider, button],])

widgets = widgetbox(slider, button)
row = row(p, l)
row_and_col = column(row, widgets)

curdoc().add_root(layout)
curdoc().title='Test Movie'


