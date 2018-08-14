import matplotlib.pyplot as plt
from customTPFs import visualize as vis

source = 219870537
dir = './figures/'

kwargs = {
    'vmin':100,
    }

test = vis(source, dir=dir)
tpf = test.tpf_movie(**kwargs)
plt.show()
