What is it?
===========

`C-Blosc2 <https://github.com/Blosc/c-blosc2>`_ is a blocking, shuffling and
lossless compression library meant for numerical data written in C.  Blosc2
is the next generation of Blosc, an
`award-winning <https://www.blosc.org/posts/prize-push-Blosc2/>`_
library that has been around for more than a decade, and that is been used
by many projects, including `PyTables <https://www.pytables.org/>`_ or
`Zarr <https://zarr.readthedocs.io/en/stable/>`_.

On top of C-Blosc2 we built Python-Blosc2, a Python wrapper that exposes the
C-Blosc2 API, plus many extensions that allow it to work transparently with
NumPy arrays, while performing advanced computations on compressed data that
can be stored either in-memory, on-disk or on the network (via the
`Caterva2 library <https://github.com/Blosc/Caterva2>`_).

Python-Blosc2 leverages both NumPy and numexpr for achieving great performance,
but with a twist. Among the main differences between the new computing engine
and NumPy or numexpr, you can find:

* Support for n-dim arrays that are compressed in-memory, on-disk or on the
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

Currently Python-Blosc2 already reproduces the API of
`Python-Blosc <https://github.com/Blosc/python-blosc>`_, so it can be
used as a drop-in replacement.  However, there are a `few exceptions
for a full compatibility.
<https://github.com/Blosc/python-blosc2/blob/main/RELEASE_NOTES.md#changes-from-python-blosc-to-python-blosc2>`_

In addition, Python-Blosc2 aims to leverage the new C-Blosc2 API so as to support
super-chunks, multi-dimensional arrays
(`NDArray <https://www.blosc.org/python-blosc2/reference/ndarray_api.html>`_),
serialization and other bells and whistles introduced in C-Blosc2.  Although
this is always and endless process, we have already catch up with most of the
C-Blosc2 API capabilities.

**Note:** Python-Blosc2 is meant to be backward compatible with Python-Blosc data.
That means that it can read data generated with Python-Blosc, but the opposite
is not true (i.e. there is no *forward* compatibility).

The main data container objects in Python-Blosc2 are:

* ``SChunk``: a 64-bit compressed store. It can be used to store any kind of data
  that supports the `buffer protocol <https://docs.python.org/3/c-api/buffer.html>`_.
* ``NDArray``: an N-Dimensional store.  This mimic the NumPy API, but with the
  added capability of storing compressed data in a more efficient way.

They are described in more detail below.

SChunk: a 64-bit compressed store
---------------------------------

``SChunk`` is the simple data container that handles setting, expanding and getting
data and metadata. Contrarily to chunks, a super-chunk can update and resize the data
that it contains, supports user metadata, and it does not have the 2 GB storage limitation.

Additionally, you can convert a SChunk into a contiguous, serialized buffer (aka
`cframe <https://github.com/Blosc/c-blosc2/blob/main/README_CFRAME_FORMAT.rst>`_)
and vice-versa; as a bonus, the serialization/deserialization process also works with NumPy
arrays and PyTorch/TensorFlow tensors at a blazing speed:

.. |compress| image:: https://github.com/Blosc/python-blosc2/blob/main/images/linspace-compress.png?raw=true
  :width: 100%
  :alt: Compression speed for different codecs

.. |decompress| image:: https://github.com/Blosc/python-blosc2/blob/main/images/linspace-decompress.png?raw=true
  :width: 100%
  :alt: Decompression speed for different codecs

+----------------+---------------+
| |compress|     | |decompress|  |
+----------------+---------------+

while reaching excellent compression ratios:

.. image:: https://github.com/Blosc/python-blosc2/blob/main/images/pack-array-cratios.png?raw=true
  :width: 75%
  :align: center
  :alt: Compression ratio for different codecs

Also, if you are a Mac M1/M2 owner, make you a favor and use its native arm64 arch (yes, we are
distributing Mac arm64 wheels too; you are welcome ;-):

.. |pack_arm| image:: https://github.com/Blosc/python-blosc2/blob/main/images/M1-i386-vs-arm64-pack.png?raw=true
  :width: 100%
  :alt: Compression speed for different codecs on Apple M1

.. |unpack_arm| image:: https://github.com/Blosc/python-blosc2/blob/main/images/M1-i386-vs-arm64-unpack.png?raw=true
  :width: 100%
  :alt: Decompression speed for different codecs on Apple M1

+------------+--------------+
| |pack_arm| | |unpack_arm| |
+------------+--------------+

Read more about ``SChunk`` features in our blog entry at: https://www.blosc.org/posts/python-blosc2-improvements

NDArray: an N-Dimensional store
-------------------------------

One of the latest and more exciting additions in Python-Blosc2 is the
`NDArray <https://www.blosc.org/python-blosc2/reference/ndarray_api.html>`_ object.
It can write and read n-dimensional datasets in an extremely efficient way thanks
to a n-dim 2-level partitioning, allowing to slice and dice arbitrary large and
compressed data in a more fine-grained way:

.. image:: https://github.com/Blosc/python-blosc2/blob/main/images/b2nd-2level-parts.png?raw=true
  :width: 75%

To wet you appetite, here it is how the ``NDArray`` object performs on getting slices
orthogonal to the different axis of a 4-dim dataset:

.. image:: https://github.com/Blosc/python-blosc2/blob/main/images/Read-Partial-Slices-B2ND.png?raw=true
  :width: 75%

We have blogged about this: https://www.blosc.org/posts/blosc2-ndim-intro

We also have a ~2 min explanatory video on `why slicing in a pineapple-style (aka double partition)
is useful <https://www.youtube.com/watch?v=LvP9zxMGBng>`_:

.. image:: https://github.com/Blosc/blogsite/blob/master/files/images/slicing-pineapple-style.png?raw=true
  :width: 50%
  :alt: Slicing a dataset in pineapple-style
  :target: https://www.youtube.com/watch?v=LvP9zxMGBng
