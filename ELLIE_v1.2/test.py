from customTPFs import custom_tpf as ctpf


ex = ctpf()
ex.pos=[250,48]
ex.multiFile='testTIC.txt'

match = ex.crossmatch_by_position(r=0.01, service='Mast.Tic.Crossmatch')
print(match)
multi = ex.crossmatch_multi_to_gaia()
print(multi)
