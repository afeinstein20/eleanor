#from postcard import postcard
#a = postcard(sector=1, camera=2, chip=1)
#a = postcard(post_name='postcard_1_3-4_2-0.fits')
#a.grab()
#print(a.local_path)
#a.read()
#print(a.xy_center)


from ffi import ffi
a = ffi(sector=1, camera=3, chip=1)
a.download_ffis()
a.sort_by_date()
a.pointing_model_per_cadence()
