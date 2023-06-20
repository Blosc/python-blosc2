What is it?
===========

`C-Blosc2 <https://github.com/Blosc/c-blosc2>`_ is the new major version of
`C-Blosc <https://github.com/Blosc/c-blosc>`_, and is backward compatible with
both the C-Blosc1 API and its in-memory format. Python-Blosc2 is a Python package
that wraps C-Blosc2, the newest version of the Blosc compressor.

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

SChunk: a 64-bit compressed store
---------------------------------

`SChunk` is the simple data container that handles setting, expanding and getting
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

Read more about `SChunk` features in our blog entry at: https://www.blosc.org/posts/python-blosc2-improvements

NDArray: an N-Dimensional store
-------------------------------

One of the latest and more exciting additions in Python-Blosc2 is the
`NDArray <https://www.blosc.org/python-blosc2/reference/ndarray_api.html>`_ object.
It can write and read n-dimensional datasets in an extremely efficient way thanks
to a n-dim 2-level partitioning, allowing to slice and dice arbitrary large and
compressed data in a more fine-grained way:

.. image:: https://github.com/Blosc/python-blosc2/blob/main/images/b2nd-2level-parts.png?raw=true
  :width: 75%

To wet you appetite, here it is how the `NDArray` object performs on getting slices
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
