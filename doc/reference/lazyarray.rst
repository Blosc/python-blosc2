.. _LazyArray:

LazyArray
=========

This is an API interface for computing an expression or a Python user defined function.

You can get an object following the LazyArray API with any of the following ways:

* Any expression that involves one or more NDArray objects. e.g. ``a + b``, where ``a`` and ``b`` are NDArray objects (see  `this tutorial <../getting_started/tutorials/03.lazyarray-expressions.html>`_).
* Using the ``lazyexpr`` constructor.
* Using the ``lazyudf`` constructor (see `a tutorial <../getting_started/tutorials/03.lazyarray-udf.html>`_).

The LazyArray object is a thin wrapper around the expression or user-defined function that allows for lazy computation. This means that the expression is not computed until the ``compute`` or ``__getitem__`` methods are called. The ``compute`` method will return a new NDArray object with the result of the expression evaluation. The ``__getitem__`` method will return an NumPy object instead.

See the `LazyExpr`_ and `LazyUDF`_ sections for more information.

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

.. _LazyExpr:

LazyExpr
--------

An expression like ``a + sum(b)``, where there is at least one NDArray object in operands ``a`` and ``b``, `returns a LazyExpr object <../getting_started/tutorials/03.lazyarray-expressions.html>`_. You can also get a LazyExpr object using the ``lazyexpr`` constructor (see below).

This object follows the `LazyArray`_ API for computation and storage.

.. autofunction:: lazyexpr

.. _LazyUDF:

LazyUDF
-------

For getting a LazyUDF object (which is LazyArray-compliant) from a user-defined Python function, you can use the lazyudf constructor below. See  `a tutorial on how this works <../getting_started/tutorials/03.lazyarray-udf.html>`_.

This object follows the `LazyArray`_ API for computation, although storage is not supported yet.

.. autofunction:: lazyudf
