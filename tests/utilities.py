import os

import blosc2


def remove_schunk(contiguous, urlpath):
    if urlpath is not None:
        if not contiguous:
            blosc2.remove_dir(urlpath)
        elif os.path.exists(urlpath):
            os.remove(urlpath)
