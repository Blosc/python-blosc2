Reduction Functions
-------------------

Contrarily to lazy functions, reduction functions are evaluated eagerly, and the result is always a NumPy array (although this can be converted internally into an :ref:`NDArray <NDArray>` if you pass any :func:`blosc2.empty` arguments in ``kwargs``).

Reduction operations can be used with any of :ref:`NDArray <NDArray>`, :ref:`C2Array <C2Array>`, :ref:`NDField <NDField>` and :ref:`LazyExpr <LazyExpr>`. Again, although these can be part of a :ref:`LazyExpr <LazyExpr>`, you must be aware that they are not lazy, but will be evaluated eagerly during the construction of a LazyExpr instance (this might change in the future).

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
