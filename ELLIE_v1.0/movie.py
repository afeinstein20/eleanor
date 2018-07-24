import numpy as np
import matplotlib.pyplot as plt
from makeLC import main, circle
from world2pix import openFITS
import matplotlib.animation as animation
from lightkurve import KeplerTargetPixelFile as ktpf


def updateFig(fn):
    mast, mheader = openFITS(fn, './calFITS_2019_3-3/')
    im = plt.imshow(mast, origin = 'lower', interpolation = 'nearest', vmin = 55, vmax = 80)
    return im

def movie():
    x, xMean, y, yMean, mast, fns, id = main()
    fig = plt.figure()
    i = 155
    pos = (x[i]-xMean[0], y[i]-yMean[0])
    ap = circle(pos, 2)
    ims = []
    for f in fns:
        im = updateFig(f)
        ims.append([im])
    ap.plot(color = 'red', alpha = 0.3, fill = 0.2)
    plt.xlim([pos[0]-5, pos[0]+5])
    plt.ylim([pos[1]-5, pos[1]+5])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    plt.show()
#    ani.save('dynamic_152.mp4')

movie()


