from ellie import data_products

#example = data_products(dir='./2019/2019_1_1-4/ffis/', camera=1, chip=4, sector=1)
example = data_products(dir='./2019/2019_1_3-3/ffis/', camera=3, chip=3, sector=1)  
#example.pointing_model()
example.make_postcard()
#example.individual_tpf()
