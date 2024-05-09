.. _LazyArray:

LazyArray API
=============

This is a class for evaluating an expression or a Python user defined function.

.. currentmodule:: blosc2.LazyArray

Methods
-------

.. autosummary::
    :toctree: autofiles/lazyarray
    :nosignatures:

    eval
    __getitem__
    save


.. _LazyExpr:

LazyExpr Usage
--------------

For getting a LazyArray from a expression, you would proceed in a similar manner
than with numexpr.

.. _LazyUDF:

LazyUDF Usage
-------------

For getting a LazyArray from a user-defined Python function, you will have to
follow the specifications in the lazyudf constructor.

.. currentmodule:: blosc2


.. autosummary::
    :toctree: autofiles/lazyarray
    :nosignatures:

    lazyudf
