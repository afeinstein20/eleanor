from customTPFs import custom_tpf as ctpf
from customTPFs import visualize as vis

ex = ctpf()
ex.pos=[250,48]
ex.multiFile='testTIC.txt'

#match = ex.crossmatch_by_position(r=0.01, service='Mast.Tic.Crossmatch')
#print(match)
#multi = ex.crossmatch_multi_to_gaia()
#print(multi)

test = vis(219870537, dir='./2019/2019_1_3-1/tic_tpf/')
lc = test.tpf_movie(cmap='Greys', com=False)
