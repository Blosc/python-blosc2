Operations with arrays
======================

Lazy Functions
--------------

The next functions can be used for computing with any of :ref:`NDArray <NDArray>`, :ref:`C2Array <C2Array>`, :ref:`NDField <NDField>` and :ref:`LazyExpr <LazyExpr>`.
The result is always a :ref:`LazyExpr` instance, which can be evaluated to get the result.

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

Reduction Functions
-------------------

Contrarily to the lazy functions above, these functions are evaluated eagerly, and the result is always a NumPy array (although this can be converted internally into an :ref:`NDArray <NDArray>` if you pass any :ref:`NDArray.empty() <NDArray.empty>` arguments in `kwargs`).

They can be used with any of :ref:`NDArray <NDArray>`, :ref:`C2Array <C2Array>`, :ref:`NDField <NDField>` and :ref:`LazyExpr <LazyExpr>`. Again, although these can be part of a :ref:`LazyExpr <LazyExpr>`, you must be aware that they are not lazy and will be evaluated eagerly during the construction of the LazyExpr instance (this might change in the future).

.. currentmodule:: blosc2

.. autosummary::
   :toctree: autofiles/operations_with_arrays/
   :nosignatures:

    all
    any
    sum
    prod
    mean
    std
    var
    min
    max
