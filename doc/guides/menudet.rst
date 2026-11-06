Menudet: portable computation
=============================

Menudet is a little language for portable computation on arrays and tables.
Saved computations run natively without reconstructing or executing Python
functions.

.. note::

   Menudet 0.1 is **beta** and is the supported language going forward. API and
   format changes remain possible and will come with migration guidance.
   For old DSL code or files, see :doc:`migrating_to_menudet`.

Getting started
---------------

Decorate a function, then call it with NumPy or Blosc2 arrays:

.. code-block:: python

    import blosc2
    import numpy as np


    @blosc2.menudet
    def adjusted(x, y):
        return x + (y - 1)


    x = np.array([1, 2, 3], dtype="float64")
    y = np.array([4, 5, 6], dtype="float64")
    values = adjusted(x, y)[:]  # NumPy values: [4., 6., 8.]
    result = adjusted(x, y).compute()  # Materialized Blosc2 NDArray.

The decorated object is a :class:`blosc2.MenudetFunction`. Calls specialize
automatically for the input dtypes and return lazy arrays. The Python body is
never executed, including after native errors.

``@blosc2.menudet`` and ``@blosc2.jit`` serve different purposes. Menudet authors
validated lazy kernels with specialization and persistence. Ordinary ``@jit``
keeps its NumPy-style eager results, tracing/control-flow dispatch, ``out=``,
``disable=`` and pandas integration. ``strict=True`` is an eager ``@jit`` parsing
contract, not an alias for Menudet. Execution policy (interpretation or JIT) is
independent of the authoring decorator.

For example, both functions compute the same values but return different kinds of
objects:

.. code-block:: python

    @blosc2.menudet
    def menudet_adjusted(x, y):
        return x + (y - 1)


    @blosc2.jit
    def jit_adjusted(x, y):
        return x + (y - 1)


    lazy = menudet_adjusted(x, y)  # Lazy result; values are not yet computed.
    values = lazy[:]  # NumPy values.
    stored = lazy.compute()  # Blosc2 NDArray, written directly into storage.

    eager = jit_adjusted(x, y)  # NumPy array, computed immediately.

Menudet's lazy result is what enables explicit specialization, partition/domain
binding and saving a validated recipe. In contrast, ``@jit`` is the convenient
route when NumPy-style eager evaluation or pandas integration is desired:

.. code-block:: python

    import pandas as pd

    frame = pd.DataFrame({"x": x, "y": y})
    doubled = blosc2.jit(lambda column: column * 2)
    eager = frame.apply(doubled, engine=blosc2.jit)  # One eager NumPy result per column.

Execution policy is independent of the authoring decorator. Without an explicit
or environment override, this single-return expression is interpreted by default;
use ``@menudet(jit=True)`` or a ``blosc2.jit_options(jit=True)`` context to prefer
JIT, and ``require_jit=True`` to require it. Interpretation is still native
Menudet execution, not evaluation of the Python body.

Types and scalar arguments
--------------------------

Array inputs supply their element dtypes and can broadcast. Numeric output
dtypes are inferred. Annotations specify element dtypes, not shapes; annotated
inputs must match exactly. Scalar parameters and nonnumeric returns require
annotations:

.. code-block:: python

    @blosc2.menudet
    def shifted(x: np.float64, offset: np.float64 = 1.0) -> np.float64:
        return x + offset


    values = shifted(x)[:]

Scalar globals and closure values are captured at decoration. Arrays, objects
and callbacks cannot be captured; pass arrays as arguments instead. See
:doc:`../reference/dsl_syntax` for supported syntax and
:doc:`../reference/menudet_accuracy` for numerical rules.

Execution policy
----------------

The default chooses native interpretation or JIT. Use ``jit=False`` for
interpretation, ``jit=True`` to prefer JIT, or ``require_jit=True`` to require it.
Complex kernels currently use the interpreter only. Computation errors propagate
without replay through Python. See :doc:`../reference/jit` for advanced controls.

Select other execution controls with ``blosc2.jit_options(...)`` when calling or
specializing a Menudet function, for example:

.. code-block:: python

    with blosc2.jit_options(jit=True, jit_backend="cc"):
        kernel = adjusted.specialize(x, y)

``jit_backend`` selects TCC or system CC; ``compiler=`` specifies the CC command,
not the backend. Existing native environment/source-pragma precedence is unchanged.
Kernel preparation fixes the execution policy; changing a context later does not
recompile an existing lazy result. Specialize/call again to select another policy.
Runtime compiler defaults are not embedded in saved artifact source. Menudet's
native floating-point policy is distinct from the expression evaluator's
``fp_accuracy`` knob; see :doc:`../reference/menudet_accuracy`.

The Menudet decorator takes kernel-authoring options, not storage controls.
Materialize with ``result.compute(cparams=..., chunks=...)`` instead. Menudet is
not a pandas engine: pass NumPy/Blosc2 arrays explicitly and materialize the lazy
result. Ordinary ``@jit`` pandas integration is unchanged.

Specialized kernels
-------------------

**MenudetFunction knows how to specialize; MenudetKernel is already specialized.**
Calling a decorated function specializes automatically for the input dtypes and
returns a lazy computation. Explicit specialization is only needed when you want
to work with the typed kernel itself.

For inspection, eager evaluation or table registration, ``specialize()`` prepares
a :class:`blosc2.MenudetKernel` without evaluating arrays:

.. code-block:: python

    kernel = adjusted.specialize(x, y)
    values = kernel.evaluate({"x": x, "y": y})

``evaluate()`` takes named NumPy inputs with matching shapes and exact dtypes.
For broadcasting and lazy evaluation, call the decorated function or use
``kernel.lazy(...)``.
When binding a specialized kernel directly, supply its logical domain/partitions
explicitly as needed; decorator partition defaults apply to decorated calls.

Input-free generators need an explicit logical domain. For example:

.. code-block:: python

    @blosc2.menudet(ndim=1, jit=False)
    def indices() -> np.int64:
        return _flat_idx


    kernel = indices.specialize()
    lazy = kernel.lazy({}, shape=(19,), partitions=(4,))
    values = lazy[:]

``ndim`` declares the native coordinate rank. ``shape`` and ``partitions`` on
the bound kernel define the domain and original logical groups; they are not
storage options. Input-free numeric 1-D generators can materialize through native
parallel workers when output blocks exactly match these partitions and chunks
contain whole blocks. Other layouts preserve groups through the general path.

Block reductions and partitions
-------------------------------

``block_sum``, ``block_prod``, ``block_min``, ``block_max``, ``block_mean``,
``block_any`` and ``block_all`` reduce elements within each logical partition.
Set ``cardinality="block_scalar"`` to return one value per partition:

.. code-block:: python

    @blosc2.menudet(cardinality="block_scalar", partitions=(3,))
    def total(x):
        return block_sum(x)


    values = total(np.arange(6, dtype="int64"))[:]
    # array([3, 12])

Partitions are independent of storage chunks and blocks. Slicing or rechunking
a result does not change its reduction groups. These operations do not accept
``axis`` or ``keepdims``; use ordinary lazy expressions for whole-array or axis
reductions.

Saving and loading
------------------

Save a lazy computation with ``save()`` and reopen it with ``blosc2.open()``.
Array operands need persistent references:

.. code-block:: python

    stored = blosc2.asarray(x, urlpath="input.b2nd", mode="w")
    lazy = adjusted(stored, stored)
    lazy.save("recipe.b2nd")
    reopened = blosc2.open("recipe.b2nd")
    values = reopened[:]

This saves the computation, captures, operand references and partitions—not its
computed values. To save values instead, use
``lazy.compute(urlpath="values.b2nd", mode="w")``.

For fixed scalar configuration in persistent recipes, use captured scalars.
Scalar parameters, including defaults, currently create temporary bindings that
cannot be saved as durable references.

Table columns
-------------

For an existing table with an ``int64`` column named ``amount``, specialize a
kernel and register its named inputs:

.. code-block:: python

    from dataclasses import dataclass


    @dataclass
    class Row:
        amount: int = 0


    table = blosc2.CTable(Row, new_data={"amount": [10, 20, 30]})


    @blosc2.menudet
    def twice(x):
        return x * 2


    kernel = twice.specialize(np.empty(0, dtype="int64"))
    table.add_computed_column("twice", kernel, inputs={"x": "amount"})
    table.add_generated_column("stored_twice", values=kernel, inputs={"x": "amount"})

Each row is independent. A fixed-shape array row can be reduced as one complete
group; reductions never combine different rows. See :doc:`../reference/ctable`.

Errors and further reading
--------------------------

Unsupported constructs and numerical errors raise rather than executing Python.
Optional raw-source validation is described in :doc:`../reference/dsl_syntax`.
Menudet is not a sandbox for arbitrary untrusted source or data.

See :doc:`../reference/menudet` for the API and :doc:`migrating_to_menudet` for
retired DSL interfaces and historical files.
