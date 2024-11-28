=============
Python-Blosc2
=============

A Python wrapper for the extremely fast Blosc2 compression library
==================================================================

:Author: The Blosc development team
:Contact: blosc@blosc.org
:Github: https://github.com/Blosc/python-blosc2
:Actions: |actions|
:PyPi: |version|
:NumFOCUS: |numfocus|
:Code of Conduct: |Contributor Covenant|

.. |version| image:: https://img.shields.io/pypi/v/blosc2.svg
        :target: https://pypi.python.org/pypi/blosc2
.. |Contributor Covenant| image:: https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg
        :target: https://github.com/Blosc/community/blob/master/code_of_conduct.md
.. |numfocus| image:: https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A
        :target: https://numfocus.org
.. |actions| image:: https://github.com/Blosc/python-blosc2/actions/workflows/build.yml/badge.svg
        :target: https://github.com/Blosc/python-blosc2/actions/workflows/build.yml


What it is
==========

`C-Blosc2 <https://github.com/Blosc/c-blosc2>`_ is a blocking, shuffling and
lossless compression library meant for numerical data written in C. On top of
it we built Python-Blosc2, a Python wrapper that exposes the C-Blosc2 API,
plus many extensions that allow it to work with NumPy arrays, while performing
advanced computations on compressed data that can be stored either in-memory,
on-disk or on the network (via the
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

You can read some of our tutorials on how to perform advanced computations at:

https://www.blosc.org/python-blosc2/getting_started/tutorials

As well as the full documentation at:

https://www.blosc.org/python-blosc2

Finally, Python-Blosc2 aims to leverage the full C-Blosc2 functionality to
support a wide range of compression and decompression needs, including
metadata, serialization and other bells and whistles.

**Note:** Blosc2 is meant to be backward compatible with Blosc(1) data.
That means that it can read data generated with Blosc, but the opposite
is not true (i.e. there is no *forward* compatibility).

NDArray: an N-Dimensional store
===============================

One of the most useful abstractions in Python-Blosc2 is the
`NDArray <https://www.blosc.org/python-blosc2/reference/ndarray_api.html>`_ object.
It enables highly efficient reading and writing of n-dimensional datasets through
a two-level n-dimensional partitioning system. This allows for more fine-grained slicing
and manipulation of arbitrarily large and compressed data:

.. image:: https://github.com/Blosc/python-blosc2/blob/main/images/b2nd-2level-parts.png?raw=true
  :width: 75%

To pique your interest, here is how the ``NDArray`` object performs when retrieving slices
orthogonal to the different axis of a 4-dimensional dataset:

.. image:: https://github.com/Blosc/python-blosc2/blob/main/images/Read-Partial-Slices-B2ND.png?raw=true
  :width: 75%

We have written a blog post on this topic:
https://www.blosc.org/posts/blosc2-ndim-intro

We also have a ~2 min explanatory video on `why slicing in a pineapple-style (aka double partition)
is useful <https://www.youtube.com/watch?v=LvP9zxMGBng>`_:

.. image:: https://github.com/Blosc/blogsite/blob/master/files/images/slicing-pineapple-style.png?raw=true
  :width: 50%
  :alt: Slicing a dataset in pineapple-style
  :target: https://www.youtube.com/watch?v=LvP9zxMGBng

Operating with NDArrays
=======================

The ``NDArray`` objects are easy to work with in Python-Blosc2.
Here it is a simple example:

.. code-block:: python

    import blosc2

    N = 20_000  # for small scenario
    # N = 50_000 # for large scenario
    a = np.linspace(0, 1, N * N).reshape(N, N)
    b = np.linspace(1, 2, N * N).reshape(N, N)
    c = np.linspace(-10, 10, N * N).reshape(N, N)
    # Expression
    expr = ((a**3 + blosc2.sin(c * 2)) < b) & (c > 0)

    # Evaluate and get a NDArray as result
    out = expr.compute()
    print(out.info)

As you can see, the ``NDArray`` instances are very similar to NumPy arrays,
but behind the scenes, they store compressed data that can be processed
efficiently using the new computing engine included in Python-Blosc2.
Although not exercised above, broadcasting and reductions also work, as well as
filtering, indexing and sorting operations for structured arrays (tables).

To pique your interest, here is the performance (measured on a modern desktop machine) that
you can achieve when the operands in the expression above fit comfortably in memory
(20_000 x 20_000):

.. image:: https://github.com/Blosc/python-blosc2/blob/main/images/lazyarray-expr.png?raw=true
  :width: 100%
  :alt: Performance when operands fit in-memory

In this case, the performance is somewhat below that of top-tier libraries like Numexpr,
but it is still quite good, specially when compared with plain NumPy.  For these short
benchmarks, numba normally loses because its relatively large compiling overhead cannot be
amortized.

One important point is that the memory consumption when using the ``LazyArray.compute()``
method is very low because the output is an ``NDArray`` object, which is compressed by default.
On the other hand, the ``LazyArray.__getitem__()`` method returns an actual NumPy array,
so it is not recommended for large datasets, as it can consume a significant amount of memory
(although it may still be convenient for small outputs, and most specially slices).

And here is the performance when the operands barely fit in memory (50_000 x 50_000):

.. image:: https://github.com/Blosc/python-blosc2/blob/main/images/lazyarray-expr-large.png?raw=true
  :width: 100%
  :alt: Performance when operands do not fit well in-memory

In this latter case, the memory consumption figures does not seem extreme, but this is because
the displayed values represent *actual* memory consumption *during* the computation
(not virtual memory); in addition, the resulting array is boolean, so it does not take too much
space to store. In this scenario, the performance compared to top-tier libraries like Numexpr
or Numba is quite competitive.  This is due to the combination of the Blosc2 compression and
the new computing engine that is able to work with compressed data very efficiently.

You can find the benchmark for the examples above at:

https://github.com/Blosc/python-blosc2/blob/main/bench/ndarray/lazyarray-expr.ipynb

https://github.com/Blosc/python-blosc2/blob/main/bench/ndarray/lazyarray-expr-large.ipynb

Installing
==========

Blosc2 now provides Python wheels for the major OS (Win, Mac and Linux) and platforms.
You can install the binary packages from PyPi using ``pip``:

.. code-block:: console

    pip install blosc2

We are in the process of releasing 3.0.0, along with wheels for various
versions.  For example, to install the first release candidate version, you can use:

.. code-block:: console

    pip install blosc2==3.0.0rc1


Documentation
=============

The documentation is available here:

https://blosc.org/python-blosc2/python-blosc2.html

Additionally, you can find some examples at:

https://github.com/Blosc/python-blosc2/tree/main/examples

Building from sources
=====================

``python-blosc2`` includes the C-Blosc2 source code and can be built in place:

.. code-block:: console

    git clone https://github.com/Blosc/python-blosc2/
    cd python-blosc2
    pip install .   # add -e for editable mode

That's it! You can now proceed to the testing section.

Testing
=======

After compiling, you can quickly verify that the package is functioning
correctly by running the tests:

.. code-block:: console

    pip install .[test]
    pytest  (add -v for verbose mode)

Benchmarking
============

If you are curious, you may want to run a small benchmark that compares a plain
NumPy array copy against compression using different compressors in your Blosc2
build:

.. code-block:: console

     python bench/pack_compress.py

License
=======

This software is licensed under a 3-Clause BSD license. A copy of the
python-blosc2 license can be found in
`LICENSE.txt <https://github.com/Blosc/python-blosc2/tree/main/LICENSE.txt>`_.

Mailing list
============

Discussion about this module are welcome on the Blosc mailing list:

blosc@googlegroups.com

https://groups.google.com/g/blosc

Mastodon
========

Please follow `@Blosc2 <https://fosstodon.org/@Blosc2>`_ to stay updated on the latest
developments.  We recently moved from Twitter to Mastodon.

Thanks
======

Blosc2 is supported by the `NumFOCUS <https://numfocus.org>`_ non for-profit
organization and `ironArray SLU <https://ironarray.io>`_, among many other
donors.

Besides the organizations above, the following people have contributed to
the core development of Blosc2:

- Francesc Alted
- Marta Iborra
- Aleix Alcacer
- Oscar Guiñon
- Ivan Vilata i Balaguer
- Oumaima Ech.Chdig

In addition, other people have contributed to the project in different
aspects:

- Jan Sellner, who contributed the mmap support for NDArray/SChunk objects.
- Dimitri Papadopoulos, who contributed a large bunch of improvements to the
  in many aspects of the project.  His attention to detail is remarkable.
- Juan David Ibáñez, who contributed different improvements.
- And many others that have contributed with bug reports, suggestions and
  improvements.

Citing Blosc
============

You can cite our work on the various libraries under the Blosc umbrella as follows:

.. code-block:: console

  @ONLINE{blosc,
    author = {{Blosc Development Team}},
    title = "{A fast, compressed and persistent data store library}",
    year = {2009-2025},
    note = {https://blosc.org}
  }

Donate
======

If you find Blosc useful and want to support its development, please consider
making a donation via the `NumFOCUS <https://numfocus.org/donate-to-blosc>`_
organization, which is a non-profit that supports many open-source projects.
Thank you!


**Make compression better!**
