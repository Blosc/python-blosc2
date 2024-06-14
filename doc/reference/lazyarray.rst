.. _LazyArray:

LazyArray API
=============

This is an interface for evaluating an expression or a Python user defined function.

You can get an object following the LazyArray API with any of the following ways:

* Any expression that involves one or more NDArray objects. e.g. ``a + b``, where
  `a` and `b` are NDArray objects.
* Using the `lazyexpr` constructor.
* Using the `lazyudf` constructor.

See the `LazyExpr`_ and `LazyUDF`_ sections for more information.

.. currentmodule:: blosc2.LazyArray

Methods
-------

.. autosummary::
    :toctree: autofiles/lazyarray
    :nosignatures:

    __getitem__
    eval
    save


.. _LazyExpr:

LazyExpr Usage
--------------

For getting a LazyArray-compliant object from an expression (à la numexpr), you can use the
lazyexpr constructor.

.. currentmodule:: blosc2

.. autosummary::
    :toctree: autofiles/lazyarray
    :nosignatures:

    lazyexpr

.. _LazyUDF:

LazyUDF Usage
-------------

For getting a LazyArray-compliant object from a user-defined Python function, you can use the
lazyudf constructor.



.. autosummary::
    :toctree: autofiles/lazyarray
    :nosignatures:

    lazyudf
