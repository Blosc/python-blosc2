.. _LazyArray:

LazyArray
=========

This is an interface for evaluating an expression or a Python user defined function.

You can get an object following the LazyArray API with any of the following ways:

* Any expression that involves one or more NDArray objects. e.g. ``a + b``, where `a` and `b` are NDArray objects (see "Computing with NDArray objects" tutorial at the :ref:`getting_started/tutorials` section).
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
