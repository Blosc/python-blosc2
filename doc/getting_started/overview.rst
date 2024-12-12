.. Try to keep in sync with the README.rst file

What is it?
===========

Python-Blosc2 is a high-performance compressed ndarray library with a flexible
compute engine.  It uses the C-Blosc2 library as the compression backend.
`C-Blosc2 <https://github.com/Blosc/c-blosc2>`_ is the next generation of
Blosc, an `award-winning <https://www.blosc.org/posts/prize-push-Blosc2/>`_
library that has been around for more than a decade, and that is been used
by many projects, including `PyTables <https://www.pytables.org/>`_ or
`Zarr <https://zarr.readthedocs.io/en/stable/>`_.

Python-Blosc2 is Python wrapper that exposes the C-Blosc2 API, *plus* a
compute engine that allow it to work transparently with NumPy arrays,
while performing advanced computations on compressed data that
can be stored either in-memory, on-disk or on the network (via the
`Caterva2 library <https://github.com/ironArray/Caterva2>`_).

Python-Blosc2 makes special emphasis on interacting well with existing
libraries and tools. In particular, it provides:

* Support for NumPy `universal functions mechanism <https://numpy.org/doc/2.1/reference/ufuncs.html>`_,
  allowing to mix and match NumPy and Blosc2 computation engines.
* Excellent integration with Numba and Cython via
  `User Defined Functions <https://www.blosc.org/python-blosc2/getting_started/tutorials/03.lazyarray-udf.html>`_.
* Lazy expressions that are computed only when needed, and that can be stored
  for later use.

Python-Blosc2 leverages both `NumPy <https://numpy.org>`_ and
`NumExpr <https://numexpr.readthedocs.io/en/latest/>`_ for achieving great
performance, but with a twist. Among the main differences between the new
computing engine and NumPy or numexpr, you can find:

* Support for ndarrays that can be compressed and stored in-memory, on-disk
  or `on the network <https://github.com/ironArray/Caterva2>`_.
* Can perform many kind of math expressions, including reductions, indexing,
  filters and more.
* Support for broadcasting operations. Allows to perform operations on arrays
  of different shapes.
* Much better adherence to the NumPy casting rules than numexpr.
* Persistent reductions where ndarrays that can be updated incrementally.
* Support for proxies that allow to work with compressed data on local or
  remote machines.

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

Operating with NDArrays
-----------------------

The ``NDArray`` objects are easy to work with in Python-Blosc2.
Here it is a simple example:

.. code-block:: python

    import blosc2

    N = 20_000  # for small scenario
    # N = 50_000 # for large scenario
    a = blosc2.linspace(0, 1, N * N).reshape(N, N)
    b = blosc2.linspace(1, 2, N * N).reshape(N, N)
    c = blosc2.linspace(-10, 10, N * N).reshape(N, N)
    # Expression
    expr = ((a**3 + blosc2.sin(c * 2)) < b) & (c > 0)

    # Evaluate and get a NDArray as result
    out = expr.compute()
    print(out.info)

As you can see, the ``NDArray`` instances are very similar to NumPy arrays,
but behind the scenes, they store compressed data that can be processed
efficiently using the new computing engine included in Python-Blosc2.

To wet your appetite, here is the performance (measured on a modern desktop machine)
that you can achieve when the operands in the expression above fit comfortably in memory
(20_000 x 20_000):

.. image:: https://github.com/Blosc/python-blosc2/blob/main/images/lazyarray-expr.png?raw=true
  :width: 90%
  :alt: Performance when operands fit in-memory

In this case, the performance is somewhat below that of top-tier libraries like
Numexpr, but still quite good, specially when compared with plain NumPy.  For
these short benchmarks, numba normally loses because its relatively large
compiling overhead cannot be amortized.

One important point is that the memory consumption when using the ``LazyArray.compute()``
method is pretty low (does not exceed 100 MB) because the output is an ``NDArray`` object,
which is compressed by default.  On the other hand, the ``LazyArray.__getitem__()`` method
returns an actual NumPy array and hence takes about 400 MB of memory (the 20_000 x 20_000
array of booleans), so using it is not recommended for large datasets, (although it may
still be convenient for small outputs, and most specially slices).

Another point is that, when using the Blosc2 engine, computation with compression is
actually faster than without it (not by a large margin, but still).  To understand why,
you may want to read `this paper <https://www.blosc.org/docs/StarvingCPUs-CISE-2010.pdf>`_.

And here it is the performance when the operands and result (50_000 x 50_000) barely fit in memory
(a machine with 64 GB of RAM, for a working set of 60 GB):

.. image:: https://github.com/Blosc/python-blosc2/blob/main/images/lazyarray-expr-large.png?raw=true
  :width: 90%
  :alt: Performance when operands do not fit well in-memory

In this latter case, the memory consumption figures do not seem extreme; this
is because the displayed values represent *actual* memory consumption *during*
the computation, and not virtual memory; in addition, the resulting array is
boolean, so it does not take too much space to store (just 2.4 GB uncompressed).

In this later scenario, the performance compared to Numexpr or Numba is quite
competitive, and actually faster than those.  This is because the Blosc2
compute engine is is able to perform the computation streaming over the
compressed chunks and blocks, for a better use of the memory and CPU caches.

You can find the notebooks for these benchmarks at:

https://github.com/Blosc/python-blosc2/blob/main/bench/ndarray/lazyarray-expr.ipynb

https://github.com/Blosc/python-blosc2/blob/main/bench/ndarray/lazyarray-expr-large.ipynb
