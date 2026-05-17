Reduction Functions
-------------------

Contrarily to lazy functions, reduction functions are evaluated eagerly, and the result is always a NumPy array (although this can be converted internally into an :ref:`NDArray <NDArray>` if you pass any :func:`blosc2.empty` arguments in ``kwargs``).

Reduction operations can be used with any of :ref:`NDArray <NDArray>`, :ref:`C2Array <C2Array>`, :ref:`NDField <NDField>` and :ref:`LazyExpr <LazyExpr>`. Again, although these can be part of a :ref:`LazyExpr <LazyExpr>`, you must be aware that they are not lazy, but will be evaluated eagerly during the construction of a LazyExpr instance (this might change in the future). When the input is a :ref:`LazyExpr`, reductions accept ``fp_accuracy`` to control floating-point accuracy, and it is forwarded to :func:`LazyExpr.compute`.

.. currentmodule:: blosc2

.. autosummary::

    all
    any
    argmax
    argmin
    count_nonzero
    cumulative_prod
    cumulative_sum
    max
    mean
    min
    prod
    std
    sum
    var



.. autofunction:: blosc2.all
.. autofunction:: blosc2.any
.. autofunction:: blosc2.argmax
.. autofunction:: blosc2.argmin
.. autofunction:: blosc2.count_nonzero
.. autofunction:: blosc2.cumulative_prod
.. autofunction:: blosc2.cumulative_sum
.. autofunction:: blosc2.max
.. autofunction:: blosc2.mean
.. autofunction:: blosc2.min
.. autofunction:: blosc2.prod
.. autofunction:: blosc2.std
.. autofunction:: blosc2.sum
.. autofunction:: blosc2.var
