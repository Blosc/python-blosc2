=============
Python-Blosc2
=============

A fast & compressed ndarray library with a flexible compute engine
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

Installing
==========

Blosc2 now provides Python wheels for the major OS (Win, Mac and Linux) and platforms.
You can install the binary packages from PyPi using ``pip``:

.. code-block:: console

    pip install blosc2

If you want to install the latest release, you can do it with pip:

.. code-block:: console

    pip install blosc2 --upgrade

For conda users, you can install the package from the conda-forge channel:

.. code-block:: console

    conda install -c conda-forge python-blosc2

Documentation
=============

The documentation is available here:

https://blosc.org/python-blosc2/python-blosc2.html

Additionally, you can find some examples at:

https://github.com/Blosc/python-blosc2/tree/main/examples

Finally, we taught a tutorial at the `PyData Global 2024 <https://pydata.org/global2024/>`_
that you can find at: https://github.com/Blosc/Python-Blosc2-3.0-tutorial.  There you will
find different Jupyter notebook that explains the main features of Python-Blosc2.

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
    pytest   # add -v for verbose mode

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

Social feeds
------------

Stay informed about the latest developments by following us in
`Mastodon <https://fosstodon.org/@Blosc2>`_,
`Bluesky <https://bsky.app/profile/blosc.org>`_ or
`LinkedIn <https://www.linkedin.com/company/88381936/admin/dashboard/>`_.

Thanks
======

Blosc2 is supported by the `NumFOCUS foundation <https://numfocus.org>`_, the
`LEAPS-INNOV project <https://www.leaps-innov.eu>`_
and `ironArray SLU <https://ironarray.io>`_, among many other donors.
This allowed the following people have contributed in an important way
to the core development of the Blosc2 library:

- Francesc Alted
- Marta Iborra
- Aleix Alcacer
- Oscar Guiñón
- Juan David Ibáñez
- Ivan Vilata i Balaguer
- Oumaima Ech.Chdig

In addition, other people have participated to the project in different
aspects:

- Jan Sellner, contributed the mmap support for NDArray/SChunk objects.
- Dimitri Papadopoulos, contributed a large bunch of improvements to the
  in many aspects of the project.  His attention to detail is remarkable.
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


**Compress Better, Compute Bigger**
