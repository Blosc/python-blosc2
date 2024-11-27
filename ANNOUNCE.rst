Announcing Python-Blosc2 3.0.0-rc1
==================================

The Blosc development team is pleased to announce the first release release of
Python-Blosc2 3.0.0. In this release, we have focused on the making of a
compute engine that can work with compressed data in a NumPy-like fashion.
You can think of Python-Blosc2 3.0 as a replacement of numexpr, but better :-)

As always, we would like to get feedback from the community before the final
release. We are providing binary wheels that you can easily install from PyPI
with:

    pip install blosc2==3.0.0rc1

For more info, you can have a look at the release notes in:

https://github.com/Blosc/python-blosc2/releases

Docs and examples are available in the documentation site:

https://www.blosc.org/python-blosc2/python-blosc2.html

What is it?
-----------

`C-Blosc2 <https://github.com/Blosc/c-blosc2>`_ is a blocking, shuffling and
lossless compression library meant for numerical data written in C. On top of
it we built Python-Blosc2, a Python wrapper that exposes the C-Blosc2 API,
plus many extensions that allow it to work with NumPy arrays, while performing
advanced computations on compressed data that can be stored either in-memory,
on-disk or on the network.

It leverages both NumPy and numexpr for achieving great performance, but with
a twist. Among the main differences between the new computing engine and NumPy
or numexpr, you can find:

* Support for ndim arrays that are compressed in-memory, on-disk or on the
  network.
* High performance compression codecs, for integer, floating point, complex
  booleans, string and structured data.
* Can perform many kind of math expressions, including reductions, indexing,
  filters and more.
* Support for NumPy ufunc mechanism, allowing to mix and match NumPy and
  Blosc2 computations.
* Excellent integration with Numba and Cython via User Defined Functions.
* Support for broadcasting operations. This is a powerful feature that
  allows to perform operations on arrays of different shapes.
* Much better adherence to the NumPy casting rules than numexpr.
* Lazy expressions that are computed only when needed, and can be stored for
  later use.
* Persistent reductions that can be updated incrementally.
* Support for proxies that allow to work with compressed data on local or
  remote machines.

You can read some of our tutorials on how to perform advanced computations at:

* https://www.blosc.org/python-blosc2/getting_started/tutorials/02.lazyarray-expressions.html
* https://www.blosc.org/python-blosc2/getting_started/tutorials/03.lazyarray-udf.html
* https://www.blosc.org/python-blosc2/getting_started/tutorials/05.persistent-reductions.html
* https://www.blosc.org/python-blosc2/getting_started/tutorials/06.remote_proxy.html

Finally, Python-Blosc2 aims to leverage the full C-Blosc2 functionality to
support a wide range of compression and decompression needs, including
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
