import numpy as np
import matplotlib.pyplot as plt
from lightkurve import KeplerTargetPixelFile as ktpf
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.widgets import RadioButtons
from removeSystematics import jitter_correction
from matplotlib.collections import PatchCollection

def click_pixels(tpf):
    coords = []
    rectList = []

    fig, ax = plt.subplots()
    ax.imshow(tpf.flux[0], origin='lower')

    def onclick(event):
        nonlocal coords, rectList
        x, y = int(np.round(event.xdata,0)), int(np.round(event.ydata,0))
        rect = Rectangle((x-0.5,y-0.5), 1.0, 1.0)
        rect.set_color('white')
        rect.set_alpha(0.3)
        rectList.append(rect)
        # Updates pixel selection right away
        ax.add_patch(rect)
        fig.canvas.draw()
        if [x,y] not in coords:
            coords.append([x,y])

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    plt.close()
    return coords, rectList
    

def check_pixels(tpf, rectList, coords):
    fig, ax = plt.subplots(1)

    ax.imshow(tpf.flux[0], origin='lower')

    for i in range(len(coords)):
        x, y = coords[i][0], coords[i][1]
        rect = Rectangle((x-0.5,y-0.5), 1.0, 1.0)
        rect.set_color('white')
        rect.set_alpha(0.3)
        ax.add_patch(rect)

    # Make Buttons              
    axRadio = plt.axes([0.05,0.5,0.15,0.2])
    butRadio = RadioButtons(axRadio, ('Yes', 'No'), activecolor='red')

    good = True
    def get_status(val):
        nonlocal good
        value = butRadio.value_selected
        if value == 'Yes':
            good = True
        else:
            good = False
    
    butRadio.on_clicked(get_status)
    plt.show()
    return good


def plot_lightcurve(tpf, coords, id, rectList):
    coords = np.array(coords)
    lc = []
    for f in range(len(tpf.flux)):
        cadence = []
        for i in range(len(coords)):
            cadence.append(tpf.flux[f][coords[i][0], coords[i][1]])
        lc.append(np.sum(cadence))
    lc = np.array(lc/np.nanmedian(lc))
    lc = jitter_correction(id, 3, 3, lc)

    plt.plot(np.arange(0,len(tpf.flux),1), lc/np.nanmedian(lc), 'k')

    plt.show()
    plt.close()
    return


def main():
    id = 198593129
    file = './figures/{}_tpf.fits'.format(id)
    tpf = ktpf.from_fits(file)
    coords, rectList = click_pixels(tpf)
    check = check_pixels(tpf, rectList, coords)
    if check == True:
        plot_lightcurve(tpf, coords, id, rectList)
    else:
        main()



main()
