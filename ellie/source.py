import numpy as np
from astropy.wcs import WCS
from astropy.table import Table

class source:
    

     def find_postcard(self):
        """
        Finds what postcard a source is located in
        Returns
        ----------
            postcard filename, header of the postcard
        """
        t = self.get_header()

        if self.pos == None and self.tic != None:
            id, pos, tmag = data_products.tic_pos_by_ID(self)
            self.pos = pos

        in_file=[None]
        # Searches through rows of the table
        for i in range(len(t)):
            data=[]
            # Creates a list of the data in a row
            for j in range(146):
                data.append(t[i][j])
            d = dict(zip(t.colnames[0:146], data))
            hdr = fits.Header(cards=d)

            xy = WCS(hdr).all_world2pix(self.pos[0], self.pos[1], 1, quiet=True)
            x_cen, y_cen, l, w = t['POST_CEN_X'][i], t['POST_CEN_Y'][i], t['POST_SIZE1'][i]/2., t['POST_SIZE2'][i]/2.
            # Checks to see if xy coordinates of source falls within postcard
            if (xy[0] >= x_cen-l) & (xy[0] <= x_cen+l) & (xy[1] >= y_cen-w) & (xy[1] <= y_cen+w):
                if in_file[0]==None:
                    in_file[0]=i
                else:
                    in_file.append(i)
                # If more than one postcard is found for a single source, choose the postcard where the
                # source is closer to the center
                if len(in_file) > 1:
                    dist1 = np.sqrt( (xy[0]-t['POST_CENX'][in_files[0]])**2 + (xy[1]-t['POST_CENY'][in_files[0]])**2  )
                    dist2 = np.sqrt( (xy[0]-t['POST_CENX'][in_files[1]])**2 + (xy[1]-t['POST_CENY'][in_files[1]])**2  )
                    if dist1 >= dist2:
                        in_file[0]=in_file[1]
                    else:
                        in_file[0]=in_file[0]
        # Returns postcard filename & postcard header
        if in_file[0]==None:
            print("Sorry! We don't have a postcard for you. Please double check your source has been observed by TESS")
            return
        else:
            self.postcard_fn = t['POST_FILE'][in_file[0]]
            return
