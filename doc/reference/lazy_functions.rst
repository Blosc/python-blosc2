.. _lazy_functions:

Lazy Functions
--------------

The next functions can be used for computing with any of :ref:`NDArray <NDArray>`, :ref:`C2Array <C2Array>`, :ref:`NDField <NDField>` and :ref:`LazyExpr <LazyExpr>`.

Their result is always a :ref:`LazyExpr` instance, which can be evaluated (with ``compute`` or ``__getitem__``) to get the actual values of the computation.

.. currentmodule:: blosc2

.. autosummary::
   :toctree: autofiles/operations_with_arrays/
   :nosignatures:

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
