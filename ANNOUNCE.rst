Announcing Python-Blosc2 2.1.0
==============================

We are happy to inform that we are introducing `NDArray`, a object for handling
multidimensional arrays using compression. It includes:
  - Data type handling (fully compatible with NumPy)
  - Double partitioning

We have plenty of documentation on `blosc2.NDArray`:
See examples at: https://github.com/Blosc/python-blosc2/tree/main/examples/ndarray
NDarray docs at: https://www.blosc.org/python-blosc2/reference/ndarray_api.html
Explanatory video on why double partitioning: https://youtu.be/LvP9zxMGBng
Also, see our blog on C-Blosc2 NDim counterpart: https://www.blosc.org/posts/blosc2-ndim-intro/

For more info, you can have a look at the release notes in:

https://github.com/Blosc/python-blosc2/releases

More docs and examples are available in the documentation site:

https://www.blosc.org/python-blosc2/python-blosc2.html


## What is it?

Python-Blosc2 is a Python package that wraps C-Blosc2, the newest version of
the Blosc compressor.  Currently Python-Blosc2 already reproduces the API of
Python-Blosc (https://github.com/Blosc/python-blosc), so the former can be
used as a drop-in replacement for the later. However, there are a few exceptions
for full compatibility:
https://github.com/Blosc/python-blosc2/blob/main/RELEASE_NOTES.md#changes-from-python-blosc-to-python-blosc2


## Sources repository

The sources and documentation are managed through github services at:

https://github.com/Blosc/python-blosc2

c-blosc2 is distributed using the BSD license, see
https://github.com/Blosc/python-blosc2/blob/main/LICENSE.txt
for details.


## Tweeter feed

Please follow @Blosc2 to get informed about the latest developments.


Enjoy Data!
- The Blosc Development Team
