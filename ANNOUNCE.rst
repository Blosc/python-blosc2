Announcing Python-Blosc2 3.0.0-rc1
==================================

The Blosc development team is pleased to announce the first release release of
Python-Blosc2 3.0.0.  Here is a summary of key achievements since the last
beta release:

**Improved Data Processing with Lazy Expressions**: Enhancements have been
made to the lazyexpr.py module, focusing on optimizing the computation of lazy
expressions over compressed multidimensional datasets. These updates streamline
operations and improve performance when working with large arrays.

**Enhanced Documentation and Tutorials**: New tutorials and updated
documentation were integrated, particularly highlighting advanced operations
like using custom user-defined functions (UDFs) with Blosc2's NDArray objects
and computing efficiently with compressed data. A big thanks to NumFOCUS for
sponsoring this.

**API Refinements**: Minor fixes and improvements to the API for consistency
and robustness were implemented, ensuring better usability for developers
working with NDArray objects and other Blosc2 features.

**Bug Fixes and Code Maintenance**: Various issues were resolved to enhance
reliability, including fixes to the Python wrapper for the Blosc2 C library.
These addressed edge cases and potential inconsistencies in handling compressed
data.

Last, but not least, we are using NumPy 2.x as the default for testing procedures
and builds. This means that our wheels are built against NumPy 2, so in case you want
to use NumPy 1.x, you will need to use NumPy 1.25.0 or later.

As always, we would like to get feedback from the community before the final release.
We are providing binary wheels that you can easily install from PyPI with:

    pip install blosc2==3.0.0rc1

For more info, you can have a look at the release notes in:

https://github.com/Blosc/python-blosc2/releases

Docs and examples are available in the documentation site:

https://www.blosc.org/python-blosc2/python-blosc2.html

What is it?
-----------

`C-Blosc2 <https://github.com/Blosc/c-blosc2>`_ is the new major version of
`C-Blosc <https://github.com/Blosc/c-blosc>`_, and is backward compatible with
both the C-Blosc1 API and its in-memory format. Python-Blosc2 is a Python
package that wraps C-Blosc2, the newest version of the Blosc compressor.

Starting with version 3.0.0, Python-Blosc2 is including a powerful computing
engine that can operate on compressed data that can be either in-memory,
on-disk or on the network. This engine also supports advanced features like
reductions, filters, user-defined functions and broadcasting.

You can read some of our tutorials on how to perform advanced computations at:

* https://github.com/Blosc/python-blosc2/blob/main/doc/getting_started/tutorials/03.lazyarray-expressions.ipynb
* https://github.com/Blosc/python-blosc2/blob/main/doc/getting_started/tutorials/03.lazyarray-udf.ipynb
* https://github.com/Blosc/python-blosc2/blob/main/doc/getting_started/tutorials/04.persistent-reductions.ipynb
* https://github.com/Blosc/python-blosc2/blob/main/doc/getting_started/tutorials/05.remote_proxy.ipynb

Python-Blosc2 aims to leverage the full C-Blosc2 functionality to
support a wide range of compression and decompression needs, but also
implementing NumPy-like multi-dimensional arrays
(`NDArray <https://www.blosc.org/python-blosc2/reference/ndarray_api.html>`_),
metadata, serialization and other bells and whistles.

**Note:** Blosc2 is meant to be backward compatible with Blosc(1) data.
That means that it can read data generated with Blosc, but the opposite
is not true (i.e. there is no *forward* compatibility).

Sources repository
------------------

The sources and documentation are managed through github services at:

https://github.com/Blosc/python-blosc2

Python-Blosc2 is distributed using the BSD license, see
https://github.com/Blosc/python-blosc2/blob/main/LICENSE.txt
for details.

Mastodon feed
-------------

Please follow https://fosstodon.org/@Blosc2 to get informed about the latest
developments.


- Blosc Development Team
  Make compression better
