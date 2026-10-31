.. _LazyArray:

LazyArray
=========

This is an API interface for computing an expression or a Python user defined function.

You can get an object following the LazyArray API in any of the following ways:

* Any expression that involves one or more NDArray objects, e.g. ``a + b``, where ``a`` and ``b`` are NDArray objects (see `the lazyarray expressions tutorial <../tutorials/02.lazyarray-expressions.html>`_).
* Using the ``lazyexpr`` constructor.
* Using the ``lazyudf`` constructor (see `a tutorial <../tutorials/03.lazyarray-udf.html>`_).
* Calling a ``@blosc2.menudet`` function (see :doc:`../guides/menudet`).

The LazyArray object is a thin wrapper around the expression or user-defined function that allows for lazy computation. This means that the expression is not computed until the ``compute`` or ``__getitem__`` methods are called. The ``compute`` method will return a new NDArray object with the result of the expression evaluation. The ``__getitem__`` method will return a NumPy object instead.

LazyArray objects also support user metadata via :attr:`LazyArray.attrs`. For
in-memory objects, this metadata lives on the Python object itself. For
persisted LazyArrays reopened from disk, metadata is synchronized with the
underlying carrier and survives reopening.

See the `LazyExpr`_ and `LazyUDF`_ sections for more information.

See :ref:`JITOptions` for ``jit``/``jit_backend`` controls, environment settings,
backend caching, and automatic interpreter fallback.

.. currentmodule:: blosc2

.. autoclass:: LazyArray
    :members:
    :inherited-members:
    :member-order: groupwise

    :Special Methods:

    .. autosummary::

        __getitem__

    Methods
    ---------------
    .. automethod:: __getitem__

    Attributes
    ----------
    .. autoattribute:: attrs

.. _LazyExpr:

LazyExpr
--------

An expression like ``a + sum(b)``, where there is at least one NDArray object in operands ``a`` and ``b``, `returns a LazyExpr object <../tutorials/02.lazyarray-expressions.html>`_. You can also get a LazyExpr object using the ``lazyexpr`` constructor (see below).

This object follows the `LazyArray`_ API for computation and storage.

For eligible elementwise expressions over matching in-memory, one-dimensional
Blosc2 operands, execution follows the requested output storage:

* ``expr.compute()`` writes directly to Blosc2 chunks through the native
  prefilter. Matching explicit ``chunks``/``blocks`` and a matching Blosc2
  ``out`` buffer remain eligible. Compression and storage options are honored;
  use ``cparams=blosc2.CParams(clevel=0)`` for uncompressed output.
* ``expr[:]`` returns NumPy values. Above the automatic parallel-map crossover,
  workers decompress and evaluate bounded tiles directly into the final NumPy
  allocation. Matching writable, contiguous NumPy ``out`` buffers can use this
  path too.

Blosc2-output requests with partial selections, incompatible grids, mixed storage
or unsupported numerical contracts retain their existing fallbacks. An execution
failure on an admitted native route is raised, rather than replayed through
another engine.

.. autofunction:: lazyexpr

.. _LazyUDF:

LazyUDF
-------

For getting a LazyUDF object (which is LazyArray-compliant) from a user-defined Python function, you can use the lazyudf constructor below. See  `a tutorial on how this works <../tutorials/03.lazyarray-udf.html>`_.

This object follows the `LazyArray`_ API for computation and storage.

.. autofunction:: lazyudf

.. _DSLKernelReference:

Legacy DSL compatibility
------------------------

The alpha ``@dsl_kernel`` decorator is retired and raises migration guidance.
Use :doc:`menudet` for supported authoring and
:doc:`../guides/migrating_to_menudet` for migration. The validators below remain
compatibility helpers; syntax acceptance alone does not certify Menudet support.

.. autofunction:: dsl_kernel

.. autofunction:: validate_dsl

.. autofunction:: validate_dsl_jit

.. autoclass:: DSLSyntaxError
