import os, sys



class postcard:

    def __init__(self, sector=None, camera=None, chip=None, post_name=None):
        if post_name != None:
            self.post_name = post_name
            self.camera = post_name[11:12]
            self.chip   = post_name[13:14]
            self.sector = post_name[9:10]
        else:
            self.camera = camera
            self.chip   = chip
            self.sector = sector

        self.post_url = 'http://jet.uchicago.edu/tess_postcards/'
        return


    def make_postcard(self, ffi_dir):
        """
        Creates 300 x 350 x n postcards, where n is the number of cadences for a given observing run
        Creates a catalog of the associated header with each postcard for use later
        """
        from astropy.io import ascii, fits
        
        if camera==None or chip==None or sector==None:
            print("You must input camera, chip, and sector you wish to create postcards for.")
            print("You can do this by calling data_products.make_postcard(sector=#, camera=#, chip=#)")
            return

        if ffi_dir==None:
            print('Please input the directory your FFIs are located in\nInitialize by calling postcard.make_postcard(ffi_dir=##)')
            return

        fns, time = self.sort_by_date(self.camera, self.chip)
        fns = [ffi_dir+i for i in fns]
        mast, mheader = fits.getdata(fns[0], header=True)
        x, y = np.linspace(45, 2093, 8, dtype=int), np.linspace(0, 2048, 8, dtype=int)
        x_cens, y_cens = [], []
        cat = 'postcard_catalog.txt'
        # Creating a table for the headers of each file
        colnames = list(mheader.keys())
        add  = ['POST_FILE', 'POST_SIZE1', 'POST_SIZE2', 'POST_CEN_X', 'POST_CEN_Y', 'POST_CEN_RA', 'POST_CEN_DEC']
        for a in add:
            colnames.append(a)

        typeList = np.full(len(colnames), 'S30', dtype='S30')
        t = Table(names=colnames, dtype=typeList)

        for i in range(len(x)-1):
            for j in range(len(y)-1):
                mast, mheader = fits.getdata(fns[0], header=True)
                fn = 'postcard_{}_{}-{}_{}-{}.fits'.format(sector, camera, chip, i, j)
                if os.path.isfile(self.post_dir+fn)==True:
                    return
                print("Creating postcard: {}".format(fn))
                x_cen = (x[i]+x[i+1]) / 2.
                y_cen = (y[j]+y[j+1]) / 2.

                radec = WCS(mheader).all_pix2world(x_cen, y_cen, 1)
                s1, s2 = 300, 350
                tpf   = ktpf.from_fits_images(images=fns, position=(x_cen,y_cen), size=(s1, s2))
                # Edits header of FITS files
                tempVals = list(mheader.values())
                moreData = [fn, s, s, x_cen, y_cen, float(radec[0]), float(radec[1])]
                for m in moreData:
                    tempVals.append(m)
                t.add_row(vals=tempVals)

                time_arrays = np.zeros((3, len(fns)))
                for f in range(len(fns)):
                    hdu = fits.open(fns[f])
                    hdr = hdu[1].header
                    time_arrays[0][f] = hdr['TSTART']
                    time_arrays[1][f] = hdr['TSTOP']
                    time_arrays[2][f] = hdr['BARYCORR']

                hdr = mheader
                hdr.append(('COMMENT', '***********************'))
                hdr.append(('COMMENT', '*     ELLIE INFO      *'))
                hdr.append(('COMMENT', '***********************'))
                hdr.append(('AUTHOR' , 'Adina D. Feinstein'))
                hdr.append(('VERSION', '1.0'))
                hdr.append(('GITHUB' , 'https://github.com/afeinstein20/ELLIE'))
                hdr.append(('CREATED', strftime('%Y-%m-%d'),
                            'ELLIE file creation date (YYY-MM-DD)'))
                hdr.append(('CEN_X'  , np.round(x_cen, 8)))
                hdr.append(('CEN_Y'  , np.round(y_cen, 8)))
                hdr.append(('CEN_RA' , float(radec[0])))
                hdr.append(('CEN_DEC', float(radec[1])))

                dtype = [
                    ("TSTART", np.float64),
                    ("TSTOP", np.float64),
                    ("BARYCORR", np.float64),
                    ("FLUX", np.float32, tpf.flux.shape[1:]),
                ]
                data = np.empty(len(tpf.flux), dtype=dtype)
                data["TSTART"] = time_arrays[0]
                data["TSTOP"] = time_arrays[1]
                data["BARYCORR"] = time_arrays[2]
                data["FLUX"] = tpf.flux

                hdu1 = fits.BinTableHDU.from_columns(fits.ColDefs(data), header=hdr)
                # hdu1 = fits.PrimaryHDU(header=hdr, data = new_postcard)
                hdu1.writeto(self.post_dir+fn)

        ascii.write(t, output='postcard_{}_{}-{}.txt'.format(sector, camera, chip))
        return


    def make_postcard_catalog(self):
        """
        Whenever a postcard is created for a camera-chip pair, a new catalog is created
            called: "postcard_{}-{}.txt".format(camera, chip)
        This function will take all of the postcard sub-catalogs and create a main one
            for each sector
        This file will be stored online and will be called from such, preventing the user
            from having to download it onto their personal machine
        Returns
        ----------
            postcard.txt: a catalog of header information for each postcard
        """
        from astropy.table import join, Table

        output_fn = 'postcard.txt'
        dir_fns = np.array(os.listdir())
        post_fns = np.array([i for i,item in enumerate(dir_fns) if 'postcard_' in item])
        post_fns = dir_fns[post_fns]
        main_table = Table.read(post_fns[0], format='ascii.basic')
        for i in range(1,len(post_fns)):
            post = Table.read(post_fns[i], format='ascii.basic')
            for j in range(len(post)):
                main_table.add_row(post[j])

        ascii.write(main_table, output='postcard.txt')
        for i in post_fns:
            os.remove(i)
        return


    def grab(self):
        """ Downloads the postcard from online """
        from astropy.utils.data import download_file

        if self.post_name==None:
            print('You have to pass in the name of the postcard you wish to grab.\nPlease initialize as postcard(post_name=###)')
            return
        else:
            print(self.post_url+self.post_name)
            local_path = download_file(self.post_url+self.post_name, cache=True)
            self.local_path = local_path
        return


    def read(self):
        """ Opens & reads postcard FITS file """
        from astropy.io import fits
        file_name = self.post_url+self.post_name
        hdu = fits.open(self.local_path)
        self.flux = hdu[0].data
#        self.flux_err = hdu[1].data
        self.header   = hdu[0].header
        self.radec_center = (self.header['CEN_RA'], self.header['CEN_DEC'])
        self.xy_center    = (self.header['CEN_X'],  self.header['CEN_Y'])
        return
