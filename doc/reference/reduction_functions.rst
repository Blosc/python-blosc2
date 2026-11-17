Reduction Functions
-------------------

Contrarily to lazy functions, reduction functions are evaluated eagerly. Full
reductions with ``keepdims=False`` return a zero-dimensional NumPy array, not a
Python or NumPy scalar, consistently across evaluation backends. Use ``result[()]``
to extract a NumPy scalar or ``result.item()`` for a Python scalar.
Partial reductions return arrays with the remaining dimensions; ``keepdims=True``
retains reduced axes with size one. Nonzero-dimensional results can be converted
internally into an :ref:`NDArray <NDArray>` by passing :func:`blosc2.empty`
arguments in ``kwargs``.

Reduction operations can be used with any of :ref:`NDArray <NDArray>`, :ref:`C2Array <C2Array>`, :ref:`NDField <NDField>` and :ref:`LazyExpr <LazyExpr>`. Again, although these can be part of a :ref:`LazyExpr <LazyExpr>`, you must be aware that they are not lazy, but will be evaluated eagerly during the construction of a LazyExpr instance (this might change in the future). When the input is a :ref:`LazyExpr`, reductions accept ``fp_accuracy`` to control floating-point accuracy, and it is forwarded to :func:`LazyExpr.compute`.

Eligible lazy-expression reductions accept ``jit`` and ``jit_backend`` through
their evaluation options. See :ref:`JITOptions`; native JIT requests remain best
effort and can use the interpreter when compilation or loading is unavailable.

When supported by the native runtime, eligible automatic whole-domain floating
``sum`` reductions over aligned one-dimensional compressed inputs, with matching
mapped and accumulator dtypes, use parallel prefilters.
Storage blocks drive decompression and expression evaluation, but
arithmetic uses fixed 128-element logical leaves and a pairwise merge tree.
Partial leaves crossing storage boundaries are completed in logical order.
For plain sums, a proven input-only map skips elementwise evaluation and the
mapped-block copy, reducing directly from the worker's decoded input buffer.
Results are independent of thread count and compression partitioning for the
same mapped values, with scratch bounded by chunk size rather than array size.
This improves accumulation accuracy without guaranteeing correctly rounded sums.
Explicit strict native execution retains its serial-order contract;
``fp_accuracy`` continues to control transcendental accuracy, not summation order.

Eligible aligned one-dimensional compressed ``min``, ``max``, ``any`` and
``all`` reductions also use parallel prefilters, bypassing proven input-only
maps. Block results are combined with the native reducer to preserve its NaN
and signed-zero rules. Partial-result storage is capped at 1 MiB; larger
domains retain the streaming bounded-reader route.

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

Grouped reductions
~~~~~~~~~~~~~~~~~~

The :func:`blosc2.group_reduce` function is a lower-level, array-oriented primitive that groups one-dimensional keys and applies eager reductions to the associated values.

.. autofunction:: blosc2.group_reduce
