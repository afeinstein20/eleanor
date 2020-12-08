import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

from .version import __version__
from .eleanor import *
from .models import * # maybe don't want to keep this 
from .prf import *
from .targetdata import *
from .postcard import *
from .source import *
from .crossmatch import *
from .ffi import *
