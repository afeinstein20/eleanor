import matplotlib.pyplot as plt
from customTPFs import visualize as vis

source = 219870537
dir = './2019/2019_1_3-1/tic_tpf/'

test = vis(source, dir=dir)
tpf = test.tpf_movie(cmap='Greys') # for Geert
plt.show()
