import numpy as np
from photutils import CircularAperture, RectangularAperture
import matplotlib.pyplot as plt
import sys, pickle

def circle(pos,r):
    return CircularAperture(pos,r)
def rectangle(pos, l, w, t):
    return RectangularAperture(pos, l, w, t)

c_list = np.array([1.25, 2.5, 3.5, 4])
r_list = np.array([3, 3, 5, 4.1])
theta = np.array([ np.pi/4., 0, np.pi/4., 0])

size = 13
center = ((size-1)/2, (size-1)/2)

d1 = (center[0]-0.5, center[1], 2, 1)
d2 = (center[0]+0.5, center[1], 2, 1)
d3 = (center[0], center[1]-0.5, 1, 2)
d4 = (center[0], center[1]+0.5, 1, 2)
di_center = np.array([d3, d2, d4, d1])
delta = 0.48
t_w = 1.6; t_l = np.sqrt(2)
t1 = (center[0]-delta, center[1]-delta, t_l, t_w, np.pi/4)
t2 = (center[0]-delta, center[1]+delta, t_w, t_l, np.pi/4)
t3 = (center[0]+delta, center[1]+delta, t_l, t_w, np.pi/4)
t4 = (center[0]+delta, center[1]-delta, t_w, t_l, np.pi/4)

t = np.array([t4, t3, t2, t1])
degrees = [0, 90, 180, 270]
types = []
all_apertures = []
for i in range(len(di_center)):
    ap  = rectangle( (di_center[i][0], di_center[i][1]), di_center[i][2], di_center[i][3], 0.0)
    tAp = rectangle( (t[i][0],t[i][1]),t[i][2], t[i][3], t[i][4])
    mask  = ap.to_mask(method='center')[0].to_image(shape=(size, size))
    tMask = tAp.to_mask(method='center')[0].to_image(shape=(size, size))

    all_apertures.append(mask)
    all_apertures.append(tMask)

    types.append('rectangle_{}'.format(degrees[i]))
    types.append('L_{}'.format(degrees[i]))


for i in range(len(r_list)):
    ap_circ = circle( center, c_list[i] )
    ap_rect = rectangle( center, r_list[i], r_list[i], theta[i])
    for method in ['center', 'exact']:
        circ_mask = ap_circ.to_mask(method=method)[0].to_image(shape=(size, size))
        rect_mask = ap_rect.to_mask(method=method)[0].to_image(shape=(size, size))
        all_apertures.append(circ_mask)
        all_apertures.append(rect_mask)
        types.append('{}_circle_{}'.format(c_list[i], method))
        types.append('{}_square_{}'.format(r_list[i], method))


dict = {}
for a in range(len(types)):
    dict[types[a]] = all_apertures[a]


fn = "default_apertures.pickle"
pickle_out = open(fn, "wb")
pickle.dump(dict, pickle_out)
pickle_out.close()

