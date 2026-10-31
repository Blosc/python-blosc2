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
