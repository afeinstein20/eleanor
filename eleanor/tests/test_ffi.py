import os.path
from ..ffi import ffi

def test_ffi_dir():
    """Are FFIs downloaded to the correct location?"""

    # set default download_dir to `~/.eleanor/sector_{}/ffis`
    default_dir = os.path.join(os.path.expanduser('~'), '.eleanor',
                               'sector_{}'.format(1), 'ffis')

    # create an ffi and fetch the download dir
    test_ffi = ffi(sector=1)
    ffi_dir = test_ffi._fetch_ffi_dir()

    # make sure the ffi_dir exists and matches the default dir
    assert(os.path.isdir(ffi_dir))
    assert(ffi_dir == default_dir)
