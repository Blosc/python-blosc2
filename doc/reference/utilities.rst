Expression Utilities
====================

A series of utilities are provided to work with expressions in a more convenient way.

.. currentmodule:: blosc2

Functions
---------
.. autofunction:: evaluate
.. autofunction:: re_evaluate
.. autofunction:: prepare_evaluate
.. autoclass:: PreparedEvaluator
   :members: evaluate
   :special-members: __call__
.. autofunction:: get_expr_operands
.. autofunction:: validate_expr

Reusable evaluation
-------------------

Prepare an operand-free recipe and supply a mapping for each execution::

    evaluator = blosc2.prepare_evaluate("x + y * 2")
    result = evaluator({"x": x, "y": y}, out=out)
    result = evaluator({"x": another_x, "y": another_y}, out=another_out)

The recipe validates the approved expression grammar at construction. Numerical
preparation is lazy and reuses the existing bounded signature caches. Shapes and
dtypes may change between calls; input and output arrays are never retained by
the recipe. Concurrent calls use independent bindings. There is no implicit
last-expression state or caller-variable capture. Per-call options override
construction options without mutating the recipe.

NumExpr-style repetition
-----------------------

For compatibility-oriented loops, ``re_evaluate()`` repeats the last successful
public ``evaluate()`` call in the current thread::

    blosc2.evaluate("x + y * 2", local_dict={"x": x, "y": y}, out=out)
    blosc2.re_evaluate(local_dict={"x": another_x, "y": another_y})

The expression and options are remembered, but operands are looked up anew.
Without dictionaries, the caller's locals/globals supply the operands. A live
explicit output is reused through a weak reference; if it has been collected,
a fresh result is allocated. Implicitly allocated results are never reused.

State is thread-local. Failed evaluations, nested evaluations and explicit
``PreparedEvaluator`` calls do not replace it. Calling ``re_evaluate()`` before
a successful ``evaluate()`` in that thread raises ``RuntimeError``. Unlike
NumExpr's unchecked shortcut, Blosc2 retains shape, dtype and destination
validation. This API is convenience compatibility, not a guarantee of lower
overhead than an explicit prepared evaluator. Use explicit evaluators for
interleaved expressions or independent callers sharing a thread.

Evaluation options
------------------

Options are validated before choosing a backend, including for Fortran outputs
and empty inputs. Unknown keywords and invalid values raise an error before
destination writes.

* ``order`` and ``casting`` describe destination allocation/conversion.
* ``optimization`` accepts ``none``, ``moderate`` and ``aggressive`` as compatibility
  hints. Neither native execution nor the NumPy fallback reproduces NumExpr's
  algebraic rewrites; NumPy-style numerical semantics remain authoritative.
* ``truediv`` accepts booleans or ``"auto"``; ordinary division retains Python 3 /
  NumPy semantics. Use ``//`` to request floor division explicitly.
* ``sanitize`` accepts a boolean or ``None``. ``True`` validates the approved
  expression grammar; ``False`` never disables safe-mode validation.
* ``disable_cache=True`` bypasses evaluate recipe reuse and native chunk
  preparation. It does not flush shared parser caches or global native caches.

Overlapping NumPy inputs are snapshotted before writing an explicit output;
this may require temporary memory proportional to the overlapping operands.
Empty sums/products use NumPy identities, and empty extrema raise without an
explicit identity. Native output descriptors support validated Fortran, transposed,
stepped and reversed destinations with bounded scratch. Unsupported operations
retain explicit NumPy fallbacks. NumExpr is not a runtime dependency;
``blosc2.numexpr`` remains a compatibility placeholder equal to ``None``.

Decorators
----------
.. autofunction:: jit
.. autofunction:: lazywhere
