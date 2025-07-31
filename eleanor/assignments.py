from astropy import units as u
from astropy.coordinates import SkyCoord
from tess_stars2px import tess_stars2px_function_entry as tess_stars2px

__all__ = ['assign_year', 'assign_target']


def assign_year(sector):
    """ Keeps track of what year each sector is observed during. """

    if sector <= 6:
        year = 2018
    elif sector <= 20:
        year = 2019
    elif sector <= 33:
        year = 2020
    elif sector <= 47:
        year = 2021
    elif sector <= 60:
        year = 2022
    elif sector <= 73:
        year = 2023
    elif sector <= 86:
        year = 2024
    elif sector <= 98:
        year = 2025
    else:
        year = 2026

    return year


def assign_target(sector):
    """ Assigns a target for a given sector to download cadences and CBVs. """

    test_coords = ['16:35:50.667 +63:54:39.87', '04:35:50.330 -64:01:37.33',
                   '04:35:50.330 -65:01:37.33', '06:00:00.000 -60:00:00.00',
                   '04:00:00.000 +10:00:00.00', '08:20:00.000 +12:00:00.00',
                   '06:00:00.000 +32:00:00.00']

    for i in range(len(test_coords)):

        coords = SkyCoord(test_coords[i], unit=(u.hourangle, u.deg))
        tess_point = tess_stars2px(0, coords.ra.deg, coords.dec.deg)
        tess_sectors = tess_point[3]
        if sector in tess_sectors:
            return coords

    return('No target assigned.')
