Announcing Python-Blosc2 2.2.7
==============================

This is a maintenance release, where we have upgraded to latest C-Blosc2 2.10.3.
This provides the ability to use the blosc2_openhtj2k dynamic plugin
(see https://github.com/Blosc/blosc2_openhtj2k for more info on this).

For more info, you can have a look at the release notes in:

https://github.com/Blosc/python-blosc2/releases

More docs and examples are available in the documentation site:

https://www.blosc.org/python-blosc2/python-blosc2.html

## What is it?

Python-Blosc2 is a Python package that wraps C-Blosc2, the newest version of
the Blosc compressor.  Currently Python-Blosc2 already reproduces the API of
Python-Blosc (https://github.com/Blosc/python-blosc), so the former can be
used as a drop-in replacement for the later. However, there are a few
exceptions for full compatibility:
https://github.com/Blosc/python-blosc2/blob/main/RELEASE_NOTES.md#changes-from-python-blosc-to-python-blosc2

Python-Blosc2 comes with NDArray support, which is a new Python class that
allows to create multi-dimensional arrays of compressed data.  NDArray
follows the same (or very similar) API as NumPy arrays, so it can be used
as a drop-in replacement.  See the documentation for more details:
https://www.blosc.org/python-blosc2/reference/ndarray_api.html

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
