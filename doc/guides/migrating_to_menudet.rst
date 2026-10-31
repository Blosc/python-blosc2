Migrating to Menudet
====================

Python-Blosc2 4.15 retires the alpha ``@dsl_kernel`` decorator and direct
``DSLKernel`` construction, including ``from_source()``. Use ``@blosc2.menudet``
for new kernels. Menudet 0.1 is beta; changes will come with migration guidance.

**Migrate historical source-backed recipes before upgrading.** They are
unsupported in this release, even with ``deserialize="full"``. The
:ref:`historical-files` section below explains how to preserve their values.

Replace the whole construction
------------------------------

Old code (no longer runnable):

.. code-block:: python

    @blosc2.dsl_kernel
    def energy(x, y):
        return x * x + y * y


    lazy = blosc2.lazyudf(energy, (x, y), dtype="float64")

Call the Menudet function directly instead:

.. code-block:: python

    import blosc2
    import numpy as np


    @blosc2.menudet
    def energy(x, y):
        return x * x + y * y


    x = np.arange(6, dtype="float64")
    y = np.ones(6, dtype="float64")
    lazy = energy(x, y)
    values = lazy[:]  # NumPy values.
    stored = lazy.compute()  # Materialized Blosc2 array.
    np.testing.assert_array_equal(values, x * x + y * y)

Passing a Menudet object to ``lazyudf()`` raises ``TypeError``. Call
``function(*inputs)`` or, for an already specialized kernel, use
``kernel.lazy({"x": x, "y": y})``.

Ordinary ``LazyExpr``/``lazyexpr()`` computations and Python ``lazyudf`` execution
remain supported. Menudet does not replace those interfaces.

Types and supported Python
--------------------------

Arrays supply their element dtypes. Annotated inputs must match exactly; scalar
parameters and nonnumeric returns require annotations:

.. code-block:: python

    @blosc2.menudet
    def shifted(x: np.float64, offset: np.float64 = 1.0) -> np.float64:
        return x + offset


    np.testing.assert_array_equal(shifted(x)[:], x + 1.0)

Scalar globals and closures are captured at decoration. Pass arrays as arguments
rather than capturing them. Callbacks, arbitrary objects, side effects such as
``print`` and unsupported dynamic Python constructs must move outside the kernel.
The Python body is never executed as a fallback.

Do not assume legacy numerical results are identical: check dtypes, conversions
and reductions against :doc:`../reference/menudet_accuracy`. See
:doc:`menudet` for execution options and :doc:`../reference/dsl_syntax` for syntax.

Reduction groups
----------------

Legacy kernels could depend on storage blocks or scalar-result broadcasting.
Menudet uses logical ``partitions`` independent of storage. A block-scalar kernel
returns one value per partition:

.. code-block:: python

    @blosc2.menudet(partitions=(3,), cardinality="block_scalar")
    def totals(x):
        return block_sum(x)


    np.testing.assert_array_equal(totals(x)[:], [3.0, 12.0])

Use ``block_*`` operations for partition reductions and ordinary lazy expressions
for whole-array or axis reductions. A decorator rename alone may not preserve
the old grouping or result shape.

Save recipes or values
----------------------

Persistent recipes require durable array operands:

.. code-block:: python

    operand = blosc2.asarray(x, urlpath="input.b2nd", mode="w")
    recipe = energy(operand, operand)
    recipe.save("energy.b2nd")
    reopened = blosc2.open("energy.b2nd")
    np.testing.assert_array_equal(reopened[:], 2.0 * x * x)

``recipe.save(...)`` preserves the computation and references.
``recipe.compute(urlpath="values.b2nd", mode="w")`` saves computed values instead.
For persistent recipes, capture fixed scalar configuration rather than using
temporary scalar arguments; see :doc:`menudet`.

.. _historical-files:

Historical files: migrate before upgrading
------------------------------------------

Legacy source recipes raise ``UnsupportedLegacyRecipeError`` under both safe and
full loading policies, including inside tables, containers and other expressions.
Ordinary arrays and supported expression-only recipes remain loadable.

1. Back up recipes, tables and operands. Keep an isolated installation of a
   previous release that supports the format until migration is verified.
2. In that **old installation**, open only trusted historical data and save
   computed values to new destinations. For example, run this in the old release:

.. code-block:: python

    old = blosc2.open("trusted_historical_recipe.b2nd", deserialize="full")
    snapshot = old.compute(urlpath="baseline_values.b2nd", mode="w")

3. For tables, export all required physical, computed and generated column values
   to ordinary arrays or a new plain physical table. Generated columns can still
   carry legacy transformer metadata: saving the old table is not enough to
   remove it. Preserve row order, dtypes, nulls and nested/variable-length schema.
4. Reauthor with Menudet in a separate candidate environment. Compare against the
   saved baseline, including boundary cases and reduction groups. Use exact
   comparisons or suitable floating-point tolerances.
5. Upgrade production only after comparison passes. Keep snapshots or save new
   Menudet recipes. Do not overwrite the originals during validation.

There is no automatic conversion or switch restoring legacy execution. Full
deserialization permission does not enable a retired format. Menudet itself is
not a sandbox for arbitrary untrusted source or data.

Table columns
-------------

Register a specialized kernel with named input bindings:

.. code-block:: python

    from dataclasses import dataclass


    @dataclass
    class Row:
        amount: float = 0.0


    @blosc2.menudet
    def twice(x):
        return x * 2.0


    table = blosc2.CTable(Row, new_data={"amount": [1.0, 2.0, 3.0]})
    kernel = twice.specialize(np.empty(0, dtype="float64"))
    table.add_computed_column("twice", kernel, inputs={"x": "amount"})
    table.add_generated_column("stored_twice", values=kernel, inputs={"x": "amount"})
    np.testing.assert_array_equal(table.twice[:], [2.0, 4.0, 6.0])

Each row is independent; reductions never combine rows. Fixed-shape array rows
can be reduced as complete groups. Historical source columns are rejected, not
silently dropped or reinterpreted. See :doc:`../reference/ctable`.

The experimental top-level ``NativeGraph`` API is also removed. Use lazy
expressions or ``@menudet`` rather than direct graph planning.
