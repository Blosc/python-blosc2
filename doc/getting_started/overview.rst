.. Try to keep in sync with the README.rst file

What is it?
===========

Python-Blosc2 is a high-performance compressed ndarray library with a flexible
compute engine.  It uses the C-Blosc2 library as the compression backend.
`C-Blosc2 <https://github.com/Blosc/c-blosc2>`_ is the next generation of
Blosc, an `award-winning <https://www.blosc.org/posts/prize-push-Blosc2/>`_
library that has been around for more than a decade, and that is being used
by many projects, including `PyTables <https://www.pytables.org/>`_ or
`Zarr <https://zarr.readthedocs.io/en/stable/>`_.

Python-Blosc2 is a Python wrapper around the C-Blosc2 library, enhanced with
an integrated compute engine. This allows for complex computations on
compressed data, whether the operands are in memory, on disk, or
`accessed over a network <https://github.com/ironArray/Caterva2>`_. This
capability makes it easier to `work with very large datasets
<https://ironarray.io/blog/compute-bigger>`_, even in distributed
environments.

Most importantly, Python-Blosc2 uses the
`C-Blosc2 simple and open format <https://github.com/Blosc/c-blosc2/blob/main/README_FORMAT.rst>`_
for storing compressed data. This facilitates seamless integration with other
systems and tools.

Interacting with the Ecosystem
==============================

Python-Blosc2 is designed to integrate seamlessly with existing libraries
and tools, offering:

* Support for NumPy's `universal functions
  mechanism <https://numpy.org/doc/2.1/reference/ufuncs.html>`_, enabling
  the combination of NumPy and Blosc2 computation engines.
* Excellent integration with Numba and Cython via
  `User Defined
  Functions <https://www.blosc.org/python-blosc2/getting_started/tutorials/03.lazyarray-udf.html>`_.
* Lazy expressions that are evaluated only when needed and can be stored
  for future use.

Python-Blosc2 leverages both `NumPy <https://numpy.org>`_ and
`NumExpr <https://numexpr.readthedocs.io/en/latest/>`_ to achieve high
performance, but with key differences. The main distinctions between the new
computing engine and NumPy or NumExpr include:

* Support for compressed ndarrays stored in memory, on disk, or
  `over the network <https://github.com/ironArray/Caterva2>`_.
* Ability to evaluate various mathematical expressions, including reductions,
  indexing, and filters.
* Support for broadcasting operations, enabling operations on arrays with
  different shapes.
* Improved adherence to NumPy casting rules compared to NumExpr.
* Support for proxies, facilitating work with compressed data on local or
  remote machines.

Data Containers
===============

The main data container objects in Python-Blosc2 are:

* ``SChunk``: A 64-bit compressed store suitable for any data type supporting the
  `buffer protocol <https://docs.python.org/3/c-api/buffer.html>`_.
* ``NDArray``: An N-Dimensional store that mirrors the NumPy API, enhanced with
  efficient compressed data storage.

These containers are described in more detail below.

SChunk: a 64-bit compressed store
---------------------------------

``SChunk`` is a simple data container that handles setting, expanding and
getting data and metadata.  In contrast to chunks, a super-chunk can update
and resize the data that it contains, supports user metadata, and has virtually
unlimited storage capacity (chunks, on the other hand, cannot store more than 2 GB).

Additionally, you can convert a SChunk into a contiguous, serialized buffer
(aka `cframe
<https://github.com/Blosc/c-blosc2/blob/main/README_CFRAME_FORMAT.rst>`_) and
vice-versa; as a bonus, the serialization/deserialization process also works
with NumPy arrays and PyTorch/TensorFlow tensors at lightning-fast speed:

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

Also, if you are a Mac M1/M2 owner, do yourself a favor and use its native arm64
arch (yes, we are distributing Mac arm64 wheels too; you're welcome ;-) ):

.. |pack_arm| image:: https://github.com/Blosc/python-blosc2/blob/main/images/M1-i386-vs-arm64-pack.png?raw=true
   :width: 100%
   :alt: Compression speed for different codecs on Apple M1

.. |unpack_arm| image:: https://github.com/Blosc/python-blosc2/blob/main/images/M1-i386-vs-arm64-unpack.png?raw=true
   :width: 100%
   :alt: Decompression speed for different codecs on Apple M1

+------------+--------------+
| |pack_arm| | |unpack_arm| |
+------------+--------------+

Read more about ``SChunk`` features in our blog entry at:
https://www.blosc.org/posts/python-blosc2-improvements

NDArray: an N-Dimensional store
-------------------------------

A recent feature in Python-Blosc2 is the
`NDArray <https://www.blosc.org/python-blosc2/reference/ndarray_api.html>`_
object.  It builds upon the ``SChunk`` object, offering a NumPy-like API
for compressed n-dimensional data.

It efficiently reads/writes n-dimensional datasets using an n-dimensional
two-level partitioning scheme, enabling fine-grained slicing of large,
compressed data:

.. image:: https://github.com/Blosc/python-blosc2/blob/main/images/b2nd-2level-parts.png?raw=true
  :width: 75%

As an example, see how the ``NDArray`` object excels at retrieving slices
orthogonal to different axes of a 4-dimensional dataset:

.. image:: https://github.com/Blosc/python-blosc2/blob/main/images/Read-Partial-Slices-B2ND.png?raw=true
  :width: 75%

More information is available in this blog post:
https://www.blosc.org/posts/blosc2-ndim-intro

Check this short video explaining `why slicing in a pineapple-style (aka
double partition) is useful
<https://www.youtube.com/watch?v=LvP9zxMGBng>`_:

.. image:: https://github.com/Blosc/blogsite/blob/master/files/images/slicing-pineapple-style.png?raw=true
  :width: 50%
  :alt: Slicing a dataset in pineapple-style
  :target: https://www.youtube.com/watch?v=LvP9zxMGBng

Operating with NDArrays
=======================

Python-Blosc2's ``NDArray`` objects are designed for ease of use,
demonstrated by this example:

.. code-block:: python

    import blosc2

    N = 20_000
    # N = 70_000 # for large scenario
    a = blosc2.linspace(0, 1, N * N).reshape(N, N)
    b = blosc2.linspace(1, 2, N * N).reshape(N, N)
    c = blosc2.linspace(-10, 10, N * N).reshape(N, N)
    expr = ((a**3 + blosc2.sin(c * 2)) < b) & (c > 0)

    out = expr.compute()
    print(out.info)

``NDArray`` instances resemble NumPy arrays but store compressed data,
processed efficiently by Python-Blosc2's engine.

When operands fit in memory (20,000 x 20,000), performance nears
top-tier libraries like NumExpr, exceeding NumPy and Numba, with low memory use
via default compression. As you can see, Blosc2 compression can speed
computation via fast codecs and filters, plus efficient CPU cache use.

.. image:: https://github.com/Blosc/python-blosc2/blob/main/images/lazyarray-dask-small.png?raw=true
  :width: 100%
  :alt: Performance when operands comfortably fit in-memory

For larger datasets exceeding memory, Python-Blosc2 rivals Dask+Zarr in
performance (70,000 x 70,000).

.. image:: https://github.com/Blosc/python-blosc2/blob/main/images/lazyarray-dask-large.png?raw=true
  :width: 100%
  :alt: Performance when operands do not fit in memory (uncompressed)

Blosc2 can utilize MKL-enabled Numexpr for optimized transcendental
functions on Intel compatible CPUs (as used for the above plots).

Benchmark notebooks:

https://github.com/Blosc/python-blosc2/blob/main/bench/ndarray/lazyarray-dask-small.ipynb

https://github.com/Blosc/python-blosc2/blob/main/bench/ndarray/lazyarray-dask-large.ipynb

Reductions and disk-based computations
--------------------------------------

One key feature of Python-Blosc2's compute engine is its ability to
perform reductions on compressed data, optionally stored on disk, enabling
calculations on datasets too large for memory.

Example:

.. code-block:: python

    import blosc2

    N = 20_000  # for small scenario
    # N = 100_000 # for large scenario
    a = blosc2.linspace(0, 1, N * N, shape=(N, N), urlpath="a.b2nd", mode="w")
    b = blosc2.linspace(1, 2, N * N, shape=(N, N), urlpath="b.b2nd", mode="w")
    c = blosc2.linspace(-10, 10, N * N, shape=(N, N))  # small and in-memory
    # Expression
    expr = np.sum(((a**3 + np.sin(a * 2)) < c) & (b > 0), axis=1)

    # Evaluate and get a NDArray as result
    out = expr.compute()
    print(out.info)

This example computes the sum of a boolean array resulting from an
expression, where the operands are on disk, with the result being a
1D array stored in memory (or optionally on disk via the ``out=``
parameter in ``compute()`` or ``sum()`` functions).

Check out a blog post about this feature, with performance comparisons, at:
https://ironarray.io/blog/compute-bigger

Hopefully, this overview has provided a good understanding of
Python-Blosc2's capabilities. To begin your journey with Python-Blosc2,
proceed to the `installation instructions <installation>`_.
Then explore the `tutorials <tutorials>`_ and
`reference <../reference>`_ sections for further information!
