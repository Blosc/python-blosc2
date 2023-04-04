Announcing Python-Blosc2 2.2.0
==============================

We have added support for bytedelta, a new filter that normally leads to better compression ratios.
We have blogged about bytedelta here: https://www.blosc.org/posts/bytedelta-enhance-compression-toolset/

We also have a short video on how bytedelta works: https://www.youtube.com/watch?v=5OXs7w2x6nw
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
