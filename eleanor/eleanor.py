import sys, os, ast
from .source import Source
from .targetdata import TargetData
from .visualize import Visualize
from .update import Update

if __name__ == "__main__":
    # Only gets run if it's called from the command line
    # Takes arguments, downloads TPF to machine
    if len(sys.argv) > 1:
        args = sys.argv

        for a in args[1::]:
            if a[0:2]=='fn':
                star = Source(fn=a[3::])
            elif a[0:6]=='coords':
                star = Source(coords=ast.literal_eval(a[7::]))
            elif a[0:3]=='gaia':
                star = Source(gaia=a[4::])
            elif a[0:3]=='tic':
                star = Source(tic=a[4::])
        data = TargetData(star)
        data.save()
    else:
        raise ValueError("To run this script, please add one of the following "
                         "arguments to your command: tic={int}, gaia={int}, "
                         "coords={list}, fn={str}.")
