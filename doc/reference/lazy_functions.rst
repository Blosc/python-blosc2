.. _lazy_functions:

Lazy Functions
--------------

The next functions can be used for computing with any of :ref:`NDArray <NDArray>`, :ref:`C2Array <C2Array>`, :ref:`NDField <NDField>` and :ref:`LazyExpr <LazyExpr>`.

Their result is always a :ref:`LazyExpr` instance, which can be evaluated (with ``compute`` or ``__getitem__``) to get the actual values of the computation.

.. currentmodule:: blosc2

.. autosummary::
    abs
    arcsin
    arccos
    arctan
    arctan2
    arcsinh
    arccosh
    arctanh
    sin
    cos
    tan
    sinh
    cosh
    tanh
    exp
    expm1
    log
    log10
    log1p
    sqrt
    conj
    real
    imag
    contains
    where

.. autofunction:: abs
.. autofunction:: arccos
.. autofunction:: arccosh
.. autofunction:: arcsin
.. autofunction:: arcsinh
.. autofunction:: arctan
.. autofunction:: arctan2
.. autofunction:: arctanh
.. autofunction:: conj
.. autofunction:: contains
.. autofunction:: cos
.. autofunction:: cosh
.. autofunction:: exp
.. autofunction:: expm1
.. autofunction:: imag
.. autofunction:: log
.. autofunction:: log10
.. autofunction:: log1p
.. autofunction:: real
.. autofunction:: sin
.. autofunction:: sinh
.. autofunction:: sqrt
.. autofunction:: tan
.. autofunction:: tanh
.. autofunction:: where
