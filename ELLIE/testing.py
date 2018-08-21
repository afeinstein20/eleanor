from customTPFs import data_products

example = data_products(dir='./2019/2019_1_3-3/ffis/', camera=3, chip=3, sector=1)
#example.pointing_model()
example.make_postcard()
