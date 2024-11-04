.. _LazyArray:

LazyArray
=========

This is an interface for evaluating an expression or a Python user defined function.

You can get an object following the LazyArray API with any of the following ways:

* Any expression that involves one or more NDArray objects. e.g. ``a + b``, where ``a`` and ``b`` are NDArray objects (see  `a tutorial <../getting_started/tutorials/03.lazyarray-expressions.html>`_).
* Using the ``lazyexpr`` constructor.
* Using the ``lazyudf`` constructor (see  `its tutorial <../getting_started/tutorials/03.LazyArray-UDF.html>`_).

See the `LazyExpr`_ and `LazyUDF`_ sections for more information.

.. currentmodule:: blosc2.LazyArray

Methods
-------

.. autosummary::
    :toctree: autofiles/lazyarray
    :nosignatures:

    __getitem__
    compute
    save


.. _LazyExpr:

LazyExpr
--------

For getting a LazyArray-compliant object from an expression (à la numexpr), you can use the lazyexpr constructor.

.. currentmodule:: blosc2

.. autosummary::
    :toctree: autofiles/lazyarray
    :nosignatures:

    lazyexpr

.. _LazyUDF:

LazyUDF
-------

For getting a LazyArray-compliant object from a user-defined Python function, you can use the lazyudf constructor.

.. autosummary::
    :toctree: autofiles/lazyarray
    :nosignatures:

    lazyudf

Utilities
---------

.. autosummary::
    :toctree: autofiles/lazyarray
    :nosignatures:

    validate_expr
    get_expr_operands
