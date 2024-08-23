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

`C-Blosc2 <https://github.com/Blosc/c-blosc2>`_ is the new major version of
`C-Blosc <https://github.com/Blosc/c-blosc>`_, and is backward compatible with
both the C-Blosc1 API and its in-memory format. Python-Blosc2 is a Python package
that wraps C-Blosc2, the newest version of the Blosc compressor.

Starting with version 3.0.0, Python-Blosc2 is including a powerful computing engine
that can operate on compressed data that can be either in-memory, on-disk or on the
network. This engine also supports advanced features like reductions, filters,
user-defined functions and broadcasting (still in beta).

You can read some of our tutorials on how to perform advanced computations at:

* https://github.com/Blosc/python-blosc2/blob/main/doc/getting_started/tutorials/03.lazyarray-expressions.ipynb
* https://github.com/Blosc/python-blosc2/blob/main/doc/getting_started/tutorials/03.lazyarray-udf.ipynb


In addition, Python-Blosc2 aims to leverage the full C-Blosc2 functionality to support
super-chunks (`SChunk <https://www.blosc.org/python-blosc2/reference/schunk_api.html>`_),
multi-dimensional arrays
(`NDArray <https://www.blosc.org/python-blosc2/reference/ndarray_api.html>`_),
metadata, serialization and other bells and whistles introduced in C-Blosc2.

**Note:** Blosc2 is meant to be backward compatible with Blosc(1) data.
That means that it can read data generated with Blosc, but the opposite
is not true (i.e. there is no *forward* compatibility).

NDArray: an N-Dimensional store
===============================

One of the more useful abstractions in Python-Blosc2 is the
`NDArray <https://www.blosc.org/python-blosc2/reference/ndarray_api.html>`_ object.
It can write and read n-dimensional datasets in an extremely efficient way thanks
to a n-dimensional 2-level partitioning, allowing to slice and dice arbitrary large and
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

Operating with NDArrays
=======================

The `NDArray` objects can be operated with very easily inside Python-Blosc2.
Here it is a simple example:

.. code-block:: python

    import numpy as np
    import blosc2

    N = 10_000
    na = np.linspace(0, 1, N * N, dtype=np.float32).reshape(N, N)
    nb = np.linspace(1, 2, N * N).reshape(N, N)
    nc = np.linspace(-10, 10, N * N).reshape(N, N)

    # Convert to blosc2
    a = blosc2.asarray(na)
    b = blosc2.asarray(nb)
    c = blosc2.asarray(nc)

    # Expression
    expr = ((a ** 3 + blosc2.sin(c * 2)) < b) & (c > 0)

    # Evaluate and get a NDArray as result
    out = expr.eval()
    print(out.info)

As you can see, the `NDArray` instances are very similar to NumPy arrays, but behind the scenes
it holds compressed data that can be operated in a very efficient way with the new computing
engine that is included in Python-Blosc2.

So as to whet your appetite, here it is the performance (with a MacBook Air M2 with 24 GB of RAM)
that you can reach when the operands fit comfortably in-memory:

.. image:: https://github.com/Blosc/python-blosc2/blob/main/images/eval-expr-full-mem-M2.png?raw=true
  :width: 100%
  :alt: Performance when operands fit in-memory

In this case, performance is a bit far from top-level libraries like Numexpr or Numba, but
it is still pretty nice (and probably using CPUs with more cores than M2 would allow closing the
performance gap even further). One important thing to know is that the memory consumption when
using the `LazyArray.eval()` method is very low, because the output is an `NDArray` object that
is compressed and in-memory by default.  On its hand `LazyArray.__getitem__()` method returns
an actual NumPy array, so it is not recommended to use it for large datasets, as it will consume
quite a bit of memory (but it can still be convenient for small outputs).

It is important to note that the `NDArray` object can use memory-mapped files as well, and the
benchmark above is actually using a memory-mapped file as the storage for the operands.
Memory-mapped files are very useful when the operands do not fit in-memory, while keeping good
performance.

And here it is the performance when the operands do not fit well in-memory:

.. image:: https://github.com/Blosc/python-blosc2/blob/main/images/eval-expr-scarce-mem-M2.png?raw=true
  :width: 100%
  :alt: Performance when operands do not fit in-memory

In the latter case, the memory consumption lines look a bit crazy, but this is because what
is displayed is the real memory consumption, not the virtual one (so, during the evaluation
the OS has to swap out some memory to disk).  In this case, the performance when compared with
top-level libraries like Numexpr or Numba is very competitive.

You can find the benchmark for the above examples at:
https://github.com/Blosc/python-blosc2/blob/main/bench/ndarray/lazyarray-expr.ipynb

Installing
==========

Blosc2 is now offering Python wheels for the main OS (Win, Mac and Linux) and platforms.
You can install binary packages from PyPi using ``pip``:

.. code-block:: console

    pip install blosc2

We are in the process of releasing 3.0.0, and we will be releasing wheels for different
beta versions.  For example, to install the first beta version, you can do:

.. code-block:: console

    pip install blosc2==3.0.0b1


Documentation
=============

The documentation is here:

https://blosc.org/python-blosc2/python-blosc2.html

Also, some examples are available on:

https://github.com/Blosc/python-blosc2/tree/main/examples


Building from sources
=====================

`python-blosc2` comes with the C-Blosc2 sources with it and can be built in-place:

.. code-block:: console

    git clone --recursive https://github.com/Blosc/python-blosc2/
    cd python-blosc2
    pip install .   # add -e for editable mode

That's all. You can proceed with testing section now.

Testing
=======

After compiling, you can quickly check that the package is sane by
running the tests:

.. code-block:: console

    python -m pip install -r requirements-tests.txt
    python -m pytest  (add -v for verbose mode)

Benchmarking
============

If curious, you may want to run a small benchmark that compares a plain
NumPy array copy against compression through different compressors in
your Blosc build:

.. code-block:: console

     PYTHONPATH=. python bench/pack_compress.py

License
=======

The software is licenses under a 3-Clause BSD license. A copy of the
python-blosc2 license can be found in
`LICENSE.txt <https://github.com/Blosc/python-blosc2/tree/main/LICENSE.txt>`_.

Mailing list
============

Discussion about this module is welcome in the Blosc list:

blosc@googlegroups.com

https://groups.google.es/group/blosc

Mastodon
========

Please follow `@Blosc2 <https://fosstodon.org/@Blosc2>`_ to get informed about the latest
developments.  We lately moved from Twitter to Mastodon.

Citing Blosc
============

You can cite our work on the different libraries under the Blosc umbrella as:

.. code-block:: console

  @ONLINE{blosc,
    author = {{Blosc Development Team}},
    title = "{A fast, compressed and persistent data store library}",
    year = {2009-2024},
    note = {https://blosc.org}
  }


----

  **Make compression better!**
