Menudet: portable computation
=============================

**Menudet is a little language for portable computation on arrays and tables.**
Python-Blosc2 provides Python-like authoring, lazy arrays and table integration;
miniexpr implements the language natively. Saved Menudet artifacts are standalone
UTF-8 JSON, not executable Python recipes. Loading one does not reconstruct or
execute a Python function.

.. note::

   Menudet 0.1 is **beta**. The core API and numerical semantics are expected to
   remain relatively stable, but may change based on feedback before the 1.0
   compatibility contract is established.

Getting started
---------------

Decorate a function with ``@blosc2.menudet``, then call it with NumPy or Blosc2
arrays. Calls return lazy arrays:

.. code-block:: python

    import blosc2
    import numpy as np


    @blosc2.menudet
    def adjusted(x, y):
        return x + (y - 1)


    x = np.array([1, 2, 3], dtype="float64")
    y = np.array([4, 5, 6], dtype="float64")
    values = adjusted(x, y)[:]  # NumPy array: [4., 6., 8.]
    result = adjusted(x, y).compute()  # Materialized Blosc2 NDArray.

The decorated object is a :class:`blosc2.MenudetFunction`. It automatically
specializes and caches executable kernels by dtype signature, not array values.
The original Python body is never executed, including when native execution fails.

Array arguments supply their element dtypes; numeric output dtypes are inferred
by the native compiler without evaluating inputs. Annotations specify element
dtypes, not shapes. Annotated arguments must match exactly, without implicit
input casts. Scalar arguments and nonnumeric returns require annotations:

.. code-block:: python

    @blosc2.menudet(jit=False)
    def shifted(x: np.float64, offset: np.float64 = 1.0) -> np.float64:
        return x + offset


    values = shifted(x)[:]

Scalar globals and closure values are snapshotted at decoration. Arrays, objects
and callbacks cannot be captured. Positional arguments, keyword arguments and
defaults are bound without executing the function. Array inputs can broadcast.

JIT and platform support
------------------------

Automatic execution selects interpretation or JIT according to the kernel and
runtime policy. Use ``jit=True`` to prefer JIT, ``jit=False`` for native
interpretation, or ``require_jit=True`` to reject kernels that cannot be compiled. Unsupported JIT
lowering can select interpretation before execution; numerical errors never
trigger replay through another backend or Python.

``complex64`` and ``complex128`` are supported by the interpreter on macOS,
Linux, Windows and WASM32. Windows builds require clang-cl. Complex kernels do
not currently use JIT. Bind complex constants as scalar captures rather than
writing ``j`` literals in native source. Complex floor division, remainder,
bitwise operations and explicit complex-to-real casts are unsupported.
Final output conversion can discard the imaginary part only with unsafe casting.

Platform/backend availability is distinct from the language's beta stability.
Floating-point status reporting is unavailable on WASM; check status support
rather than assuming zero reported flags means exception-free computation.

Kernels and artifacts
---------------------

Most users do not need to specialize explicitly. For export or inspection,
``specialize()`` returns a :class:`blosc2.MenudetKernel` without evaluating arrays:

.. code-block:: python

    kernel = adjusted.specialize(x, y)
    artifact = kernel.to_json()
    imported = blosc2.MenudetKernel.from_json(artifact, jit=False)
    values = imported.evaluate({"x": x, "y": y})  # Eager NumPy result.

Existing :class:`blosc2.DSLKernel` authoring can also export Menudet artifacts:

.. code-block:: python

    author = blosc2.DSLKernel.from_source("def affine(x):\n    return x * 2.0 - 1.0\n")
    artifact = author.export({"x": "float64"}, "float64")
    kernel = blosc2.MenudetKernel.from_json(artifact)
    values = kernel.evaluate({"x": np.arange(4, dtype="float64")})
    # array([-1., 1., 3., 5.])

Runtime bindings are named and require exact dtypes. The eager ``evaluate`` API
requires matching shapes; ``lazy`` provides logical broadcasting. Artifacts
record input/output signatures, scalar captures, numerical policies and logical
context. Version 0.1 is the only supported artifact version.

Language and numerical behavior
-------------------------------

Menudet supports Boolean values, standard signed and unsigned integer widths,
``float32``, ``float64``, complex values and fixed-width bytes/Unicode strings.
Core constructs include arithmetic, comparisons, Boolean logic, local variables,
branches, bounded loops, mathematical functions and explicit dtype conversions.
Objects, callbacks and unsupported dynamic indexing are rejected. See
:doc:`dsl_syntax` for authoring syntax and :doc:`menudet_accuracy` for numerical
accuracy and its limits.

Important numerical rules:

* Strong typed integer arithmetic wraps at its computation width. Weak source
  literals and ordinary Python scalar captures follow NumPy 2.x scalar-strength
  rules; weak scalar construction and integer block accumulation are checked.
* Output conversion happens after computation; requesting a wider output does
  not widen intermediate arithmetic. ``casting`` is ``"unsafe"`` by default;
  ``"same_kind"`` and ``"safe"`` are available.
* Floating-to-integer conversions truncate but reject nonfinite or out-of-range
  values, rather than relying on platform-dependent conversion sentinels.
* ``where`` evaluates only selected lanes. NaN and signed-zero behavior follows
  Menudet's defined rules; bitwise agreement with every NumPy loop is not promised.
* Reduction grouping can differ from NumPy. Parallel floating sums can also
  differ with thread count because addition order changes.

For eager kernel calls, ``return_status=True`` returns floating-point status
alongside values; ``fp_errors="raise"`` raises :class:`blosc2.MenudetError` with
``fp_status`` attached. The default is ``fp_errors="ignore"``. Status bits are
invalid=1, divide-by-zero=2, overflow=4 and underflow=8. NumPy warning/callback
emulation is not provided.

Block reductions and partitions
-------------------------------

``block_sum``, ``block_prod``, ``block_min``, ``block_max``, ``block_mean``,
``block_any`` and ``block_all`` reduce the valid, participating lanes of one
logical evaluation block. They do not accept ``axis`` or ``keepdims`` and do not
combine results across blocks. Ordinary lazy expressions and native graphs have
separate whole-array and axis reduction APIs.

.. code-block:: python

    author = blosc2.DSLKernel.from_source("def total(x):\n    return block_sum(x)\n")
    artifact = author.export({"x": "int64"}, "int64", cardinality="block_scalar")
    kernel = blosc2.MenudetKernel.from_json(artifact)
    values = kernel.lazy({"x": np.arange(6, dtype="int64")}, partitions=(3,))[:]
    # array([3, 12])

Partitions define an immutable logical C-order grid, independent of compressed
storage chunks and blocks. Elementwise results retain the logical domain;
block-scalar results contain one item per partition. Slices evaluate original
complete groups before selecting result lanes. Rechunking output never changes
reduction membership. Coordinate-bearing sources require a covering ``ndim``.

Integer block sums/products use fixed 64-bit accumulators and raise on overflow.
Integer means use checked integer accumulation and return float64. Empty sums,
products and truth reductions have their defined identities; empty means return
NaN, while extrema reject empty participating groups.

Saving and loading
------------------

Lazy Menudet results can persist their artifacts, bindings and logical partitions.
Operands must have persistent references:

.. code-block:: python

    stored = blosc2.asarray(x, urlpath="input.b2nd", mode="w")
    lazy = shifted(stored)
    lazy.save("recipe.b2nd")
    reopened = blosc2.open("recipe.b2nd")
    values = reopened[:]

Menudet recipes load without reconstructing Python functions. Unsupported recipes
fail validation before destination writes. Historical Python-source recipes are
a separate feature and require explicit ``deserialize="full"`` for trusted data;
see :doc:`save_load` for deserialization policies.

Table integration
-----------------

Computed and generated columns accept Menudet kernels with named input bindings:

.. code-block:: python

    author = blosc2.DSLKernel.from_source("def twice(x):\n    return x * 2\n")
    kernel = blosc2.MenudetKernel.from_json(author.export({"x": "int64"}, "int64"))
    table.add_computed_column("twice", kernel, inputs={"x": "amount"})
    table.add_generated_column("stored_twice", values=kernel, inputs={"x": "amount"})

Registration uses independent-row semantics, not table-wide or storage-block
reduction membership. A scalar row is a one-lane group; a fixed-shape array row
can be reduced as one complete group with a block-scalar kernel. Incompatible
shapes, cardinalities and dtypes reject before registration. See :doc:`ctable`.

Validation and errors
---------------------

:func:`blosc2.validate_menudet` checks raw native source and a typed signature
without executing it or requesting JIT:

.. code-block:: python

    info = blosc2.validate_menudet(
        "def increment(x):\n    return x + 1\n", {"x": "int64"}, "int64"
    )
    assert info["valid"]

The validator accepts native syntax, not arbitrary Python normalization or live
globals. Import/export errors raise :class:`blosc2.MenudetError` with a ``status``
attribute and available native source diagnostics. Static validation cannot prove
that every input stays within runtime numerical domains. Loop limits, invalid
casts and other evaluation errors still propagate.

.. warning::

   Menudet is not a sandbox for untrusted source or arbitrary data.

Advanced execution
------------------

``MenudetKernel.evaluate_array`` provides logical broadcasting and axis reductions;
``evaluate_sources`` exposes bounded compressed-source reads and scratch reports.
Experimental :class:`blosc2.NativeGraph` plans support array expressions and
reductions. Iterator scratch reports are not total-memory limits: output storage,
decoder caches and normalization copies are separate.

The authoritative language and artifact specifications live in miniexpr under
``doc/dsl-spec/0.1.md`` and ``doc/dsl-spec/artifact-0.1.md``.

API
---

.. autofunction:: blosc2.menudet

.. autoclass:: blosc2.MenudetFunction
    :members:

.. autoclass:: blosc2.MenudetKernel
    :members:

.. autoclass:: blosc2.MenudetError

.. autofunction:: blosc2.validate_menudet

.. autoclass:: blosc2.NativeGraph
    :members:
