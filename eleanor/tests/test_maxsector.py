import datetime
from ..maxsector import max_sector

def test_max_sector():
    test_date = datetime.datetime(2026, 7, 16)
    mx = max_sector(test_date)
    assert(mx==105)

def test_future_max_sector():
    test_date = datetime.datetime(2027, 1, 19)
    mx = max_sector(test_date)
    assert(mx==112)
