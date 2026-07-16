import datetime
import numpy as np

__all__ = ['set_dt', 'upcoming_sectors', 'max_sector']


def set_dt(yr, mn, dy):
    return datetime.datetime(yr, mn, dy)


def upcoming_sectors():
    """
    Sets the end date for upcoming sectors from https://tess.mit.edu/. This
    will be updated once future observing sector dates become available.

    Returns
    -------
    new_sectors : dict
       A dictionary of the end dates for the upcoming sectors available on the
       TESS MIT website.
    """
    new_sectors = {set_dt(2026,7,11):  105,
                   set_dt(2026,8,9):   106,
                   set_dt(2026,9,7):   107,
                   set_dt(2026,10,5):  108,
                   set_dt(2026,10,31): 109,
                   set_dt(2026,11,26): 110,
                   set_dt(2026,12,22): 111,
                   set_dt(2027,1,17):  112,
                   set_dt(2027,2,13):  113,
                   set_dt(2027,3,14):  114,
                   set_dt(2027,4,12):  115,
                   set_dt(2027,5,10):  116,
                   set_dt(2027,6,7):   117,
                   set_dt(2027,7,3):   118,
                   set_dt(2027,7,29):  119,
                   set_dt(2027,8,23):  120
    }
    return new_sectors


def max_sector(date=None):
    """
    Automatically determines the max sector from the available survey strategy.
    """
    if date == None:
        date = datetime.datetime.now()

    available_sectors = upcoming_sectors()
    sector_end_dates  = np.array(list(available_sectors.keys()))

    latest = np.where(sector_end_dates < date)[0][-1]
    max_sector = available_sectors[sector_end_dates[latest]]

    return max_sector
