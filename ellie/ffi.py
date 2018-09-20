class ffi:

    def __init__(self, sector=None, camera=None, chip=None):
        self.sector = sector
        self.camera = camera
        self.chip   = chip


    def download_ffis(self, sector=None, camera=None, chips=None):
        """ Downloads entire sector of data into .ellie/ffis/sector directory """
        def findAllFFIs(ca, ch):
            nonlocal year, days, url
            calFiles = []
            for d in days:
                path = '/'.join(str(e) for e in [url, year, d, ca])+'-'+str(ch)+'/'
                for fn in BeautifulSoup(requests.get(path).text, "lxml").find_all('a'):
                    if fn.get('href')[-7::] == 'ic.fits':
                        calFiles.append(path+fn.get('href'))
            return calFiles

        for s in sector:
            ffi_dir, sect_dir = 'ffis', 'sector_{}'.format(sector)
            # Creates an FFI directory, if one does not already exist
            if os.path.isdir(self.root_dir+'/'+sect_dir) == False:
                os.system('cd {} && mkdir {}'.format(self.root_dir, sect_dir))
            # Creates a sector directory, if one does not already exist
            if os.path.isdir(self.root_dir+'/'+sect_dir+'/'+ffi_dir) == False:
                os.system('cd {} && mkdir {}'.format(self.root_dir+'/'+sect_dir, ffi_dir))

            self.sect_dir = '/'.join(str(e) for e in [self.root_dir, sect_dir, ffi_dir])

            if sector in np.arange(1,14,1):
                year=2019
            else:
                year=2020
            # Current days available for ETE-6
            days = np.arange(129,132,1)
            if camera==None:
                camera = np.arange(1,5,1)
            if chips==None:
                chips  = np.arange(1,5,1)
            # This URL applies to ETE-6 simulated data ONLY
            url = 'https://archive.stsci.edu/missions/tess/ete-6/ffi/'
            for c in camera:
                for h in chips:
                    files = findAllFFIs(c, h)
                    # Loops through all files from MAST
                    for f in files:
                        file = Path(self.sect_dir+f)
                        # If the file in that directory doesn't exist, download it
                        if file.is_file() == False:
                            os.system('cd {} && curl -O -L {}'.format(self.sect_dir, f))
        return


     def sort_by_date(self, camera, chip):
        """ Sorts FITS files by start date of observation """
        fns = np.array(os.listdir(self.ffi_dir))
        pair = '{}-{}'.format(camera, chip)
        ffisInd = np.array([i for i,item in enumerate(fns) if 'ffic' and pair in item])
        camchip = np.array([i for i,item in enumerate(fns) if pair in item])
        fitsInd = [i for i in ffisInd if i in camchip]

        fns = fns[fitsInd]
        dates, time = [], []
        for f in fns:
            mast, header = fits.getdata(self.ffi_dir+f, header=True)
            dates.append(header['DATE-OBS'])
            time.append((header['TSTOP']-header['TSTART'])/2.)
        dates, fns = np.sort(np.array([dates, fns]))
        dates, time = np.sort(np.array([dates, time]))
        return fns, time

