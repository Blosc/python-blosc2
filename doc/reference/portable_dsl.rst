Menudet: portable computation
=============================

**Menudet — a little language for portable computation on arrays and tables.**

Menudet is the language; miniexpr is its native implementation, and Python-Blosc2
provides authoring, array scheduling and table integration. The existing
``DSLKernel``, ``PortableKernel`` and ``validate_portable_dsl`` API names remain
unchanged. Python-like authoring syntax normalizes to a language-independent
artifact; a Menudet artifact is not an executable Python recipe.

.. warning::

   Menudet 1.0 is experimental. Its language and artifact specifications are
   drafts, not a frozen compatibility promise. Native cross-platform CI passes;
   independent numeric accuracy and Python release-wheel certification are
   tracked separately. Beta feedback will inform the final compatibility contract.

Menudet artifacts use the versioned native 1.0 profile.
Artifacts are standalone UTF-8 JSON, executable through native miniexpr in C or
Python, without reconstructing or executing Python functions on import.

Opt-in NumPy arithmetic profile 1.1
----------------------------------

An updated miniexpr runtime additionally supports an experimental ``1.1``
language/artifact pair. It does **not** reinterpret saved 1.0 artifacts: those
retain checked arithmetic, and export still defaults to 1.0. Older native
dependencies reject 1.1 explicitly.

Inverse functions use canonical Array API/C names: ``asin``, ``acos``, ``atan``,
``atan2``, ``asinh``, ``acosh`` and ``atanh``. Python authoring accepts NumPy's
``arc*`` spellings and normalizes calls before native compilation/export. Native
source and raw saved artifacts containing these removed aliases reject in both
profiles; update the source or re-export from authoring. Artifact import does not
silently migrate saved source.

.. code-block:: python

    import blosc2
    import numpy as np

    author = blosc2.DSLKernel.from_source("def increment(x):\n    return x + 1\n")
    artifact = author.export({"x": "int8"}, "int64", version="1.1", casting="safe")
    kernel = blosc2.PortableKernel.from_json(artifact)
    kernel.inferred_dtype  # dtype('int8'), before final int64 conversion
    kernel.evaluate({"x": np.array([127, -128, 0], dtype="int8")})
    # array([-128, -127, 1], dtype=int64): arithmetic wraps at int8, not int64

Profile 1.1 implements fixed-width wrapping arithmetic, NumPy strong dtype
promotion, Boolean arithmetic restrictions, floor division/remainder, bitwise
operators and shifts. Weak source literals and plain Python numeric captures
preserve NumPy 2.x scalar strength; NumPy scalar captures and explicit
``capture_dtypes`` are strong. Typed runtime inputs, including 0-D arrays,
remain strong. Scalar categories and numerical policy are stored in the artifact.

Fixed-width casts such as ``int8(x)`` and ``float32(x)`` are native 1.1 syntax.
Final output conversion declares ``casting="safe"``, ``"same_kind"`` or
``"unsafe"``; inference and policy validation inspect metadata, not input values.
Weak scalar construction is checked; unsafe array integer narrowing wraps.
Floating-to-integer conversion truncates but deliberately rejects nonfinite or
out-of-range values instead of codifying NumPy's platform-dependent sentinels.
Function signatures and exceptional values are covered by the native M4 matrix.
``minimum``/``maximum`` propagate NaNs while ``fmin``/``fmax`` select a non-NaN
operand. Signed-zero ties follow deterministic IEEE minimum/maximum rules;
NumPy's tie bits can vary between loops. ``where`` evaluates only selected lanes,
unlike eager NumPy argument evaluation. Float16-result signatures reject explicitly.

Profile 1.1 also provides per-call IEEE floating status:

.. code-block:: python

    inputs = {"x": np.array([127, 0], dtype="int8")}
    values, status = kernel.evaluate(inputs, return_status=True)
    # status = {"flags": integer_bits, "supported": bool}
    values = kernel.evaluate(inputs, fp_errors="raise")

Bits are invalid=1, divide-by-zero=2, overflow=4 and underflow=8. Each call clears
its status; standalone block users explicitly OR status across blocks or threads.
Only evaluated active operations participate, and native calls restore the caller's
rounding mode and flags. ``fp_errors="ignore"`` is default; ``"raise"`` attaches
``.fp_status`` to ``PortableArtifactError``. No NumPy warning/callback emulation
is promised. On WASM, status reports ``supported=False`` and raising is unsupported;
zero flags there do not imply exception-free computation.

Logical arrays and native-required graphs
----------------------------------------

The experimental native array ABI adds logical broadcasting and axis reductions
without changing explicit block-reduction recipes:

.. code-block:: python

    # A numeric, elementwise 1.1 kernel, with rank-zero logical context:
    values, report = kernel.evaluate_array(inputs, return_report=True)
    totals = kernel.evaluate_array(inputs, reduction="sum", axis=-1, keepdims=True)

Inputs may be C/F-order, transposed, negative-stride, unaligned or byte-swapped
views, and singleton/0-D inputs broadcast to the logical domain. Output owns
C-order storage. Native iterator scratch is bounded by ``tile_items`` and reported
separately from cumulative gathers and host normalization copies. In-place/aliased
output is unsupported. Scalar initial values and broadcast Boolean participating
masks are supported by the six reductions: sum/prod/min/max/any/all.

Reduction grouping is serial logical C order, independent of iterator tiles and
storage chunks; floating sums need not match NumPy's pairwise bits. Narrow integer
accumulators wrap modularly. Floating-to-integer accumulator overrides, cumulative
and arg reductions, general gathers/scatters and mutable views are unsupported.

Integer sum/product reductions default to fixed 64-bit accumulators: ``int64``
for Boolean and signed integer inputs, and ``uint64`` for unsigned inputs.
This preserves artifact semantics across platforms and matches NumPy's defaults
on 64-bit hosts, but not its platform-dependent defaults on 32-bit hosts such as
WASM32. On those hosts, result dtypes and overflow behavior can differ from NumPy.

An explicit experimental graph execution mode rejects rather than falling back:

.. code-block:: python

    expr = blosc2.lazyexpr("x * 2 + 1", {"x": array})
    result = expr.compute(_require_native=True)
    accelerated = expr.compute(_require_native=True, jit=True)
    totals = expr.sum(axis=0, _require_native=True)
    artifact = expr.native_kernel().to_json()

Eligible arithmetic/functions fuse into an immutable native graph plan. Python
adapts syntax, explicit captures and storage owners; miniexpr owns numerical
validation, type inference, broadcasting, reduction planning and execution.
Python does not generate DSL/export twice or plan broadcasting on this route.
Plans cache signatures/captures, never
operand results. NumExpr is not needed for this subset, but its packaging dependency
has not been relaxed. Backend defaults are unchanged. Nested lazy/proxy/table
operands, row filtering/ordering, output aliases, reduction partial reads and
acceleration overrides reject. Basic elementwise slicing is supported.

Native iterator scratch is not an end-to-end memory bound: compressed operands
can still require full frontend materialization. The execution report records those
bytes separately. Exported elementwise portable artifacts can use existing safe
portable persistence; logical reduction descriptors are not persisted as though
they were block-scalar recipes.

Native declarative preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Graph-enabled miniexpr builds additionally expose ``blosc2.NativeGraph``. This
experimental format (``menudet-graph-1``) is separate from artifact versions and
portable recipe persistence. Preparation never requires numerical input buffers:

.. code-block:: python

    plan = blosc2.NativeGraph.from_expression(
        "where(x != 0, y / x, y)", {"x": "float32", "y": "float32"}
    )
    schedule = plan.specialize({"x": ("float32", (3,)), "y": ("float32", ())})
    metadata = schedule.info()  # Native map/output shapes and result dtype.
    values, report = schedule.execute(
        {
            "x": np.array([0, 2, 4], dtype="float32"),
            "y": np.array(8, dtype="float32"),
        }
    )
    imported = blosc2.NativeGraph.from_json(plan.to_json())

The restricted native text grammar is not general Python. Declarative JSON stores
explicit weak/typed categories, lossless scalar encodings and optional root
sum/prod/min/max/any/all options. Native final conversion is distinct from map
precision/reduction accumulation and has an explicit intermediate memory budget.
Input shapes/dtypes must match the schedule; re-specialize changed shapes, not
changed values. Schedules retain their immutable plans and permit independent
concurrent invocations. Actual JIT selection is reported by ``plan.has_jit``.

Intermediate reductions and shared array computations use the separately declared
materialized staged subset. Native text and Python adapters declare the staged
capability when they contain intermediate reductions; miniexpr alone determines
the stage boundaries and inferred dependency types. For example:

.. code-block:: python

    plan = blosc2.NativeGraph.from_expression("x - sum(x, axis=0)", {"x": "float32"})
    schedule = plan.specialize({"x": ("float32", (3, 4))}, intermediate_budget=16)
    stage = schedule.stage_info(0)  # Output shape/dtype/bytes and last consumer.

All intermediate buffers are budgeted/preallocated before any stage executes and
released after their last consumer. Reports separate iterator scratch, reserved
intermediate bytes and actual interpreted/JIT stage counts. Explicit staged JSON
may also compose trusted, validated portable 1.1 elementwise/context-free regions
with an ordered-lazy, unmasked contract; arbitrary full-DSL recipes still reject.
Implicit conditional/masked staging, scalar-only weak stage sharing, computed
participation masks, callbacks, storage IO and partial staged reads remain excluded.
Older native
dependencies reject graph preparation explicitly, without retaining a duplicate
Python planner. For exact-pair local builds, pass
``-Ccmake.define.FETCHCONTENT_SOURCE_DIR_MINIEXPR=/path/to/miniexpr`` to pip.

An explicit ``PortableKernel.from_json(artifact, jit=True)`` request can accelerate
the supported 1.1 subset with TCC or the system C compiler: numeric elementwise
expressions, exact integral comparisons, lazy ``where`` and Boolean expressions,
locals, conditional returns, logical ND context, and lane-local ``for``/``while``
loops with nested ``break``/``continue`` and returns. Range conversion and while
limits retain their checked semantics. Strong typed integer arithmetic remains
modular; weak integer arithmetic remains checked.

Checked numeric casts and numeric builtins, including integer-preserving math,
combinatorial functions, ``ldexp`` and ``fma``, use exact-bit scalar bridges where
direct lowering is unavailable. These bridges execute individual operations on
already computed operands, not entire expression subtrees. Computed weak operands
and floating captures can use runtime checks; validated immutable integer/Boolean
capture leaves retain their value-specialized fast path.

A single block-scalar return of float32/float64 ``block_sum`` or ``block_prod`` can
also compile, including an eligible mapped operand such as ``block_sum(x * 2)``.
Its loop fuses the map with serial accumulation, preserving per-operation rounding,
participating masks, empty-group identities and original logical partitions.
This is not parallel reduction or streaming of staged graph intermediates.
Check ``kernel.has_jit`` to distinguish compiled execution from fallback. Select
the backend before importing the artifact with ``ME_DSL_JIT_COMPILER=tcc`` or
``ME_DSL_JIT_COMPILER=cc``; ``CC`` selects the latter compiler (e.g. GCC).

Participating masks and floating diagnostics retain portable semantics. Integral
comparisons never pass through floating-point transport; floating comparisons use
a host bridge for NaN exception behavior. Ordered temporaries and immediate error
checks prevent failed operands from evaluating later operands. Local stores retain
assignment precision and diagnostics even if their values are unused.

Checked integer reductions, mean/extrema/truth reductions, multiple reductions and
scalar statement programs retain interpreter/native-helper routes. Unsupported
definite-assignment patterns, fixed strings, and backend limitations also fall back.
Portable WASM JIT is not enabled: the legacy WASM source adapter does not preserve
the exact 64-bit typed-bridge ABI. Passing WASM interpretation tests is not JIT
qualification.
Arbitrary ``CFLAGS``/TCC option overrides make this subset ineligible rather
than weakening strict floating semantics. No explicit SIMD qualification is claimed.
Checked 1.0 and default backend selection remain unchanged; profile and scalar
categories survive export/import and lazy recipe persistence.

Default checked profile 1.0
---------------------------

.. code-block:: python

    import blosc2
    import numpy as np

    author = blosc2.DSLKernel.from_source("def affine(x):\n    return x * 2.0 - 1.0\n")
    artifact = author.export({"x": "float64"}, "float64")
    kernel = blosc2.PortableKernel.from_json(artifact, jit=False)
    values = kernel.evaluate({"x": np.arange(4, dtype="float64")})
    # array([-1., 1., 3., 5.])

The result is an eager NumPy array. Bindings are named and require exact dtypes
and matching shapes. Supported dtypes are Boolean, standard signed/unsigned
integer widths, ``float32``, ``float64`` and fixed-width bytes/Unicode strings;
parameters may have different dtypes. Captured Python/NumPy scalars are snapshotted into
typed constants, never live callbacks or arrays. Integers use range-checked
decimal strings; floats use exact IEEE-754 hexadecimal bits.

For a runnable array/table example using the same kernel, see
``examples/menudet.py`` in the source distribution. Numeric certification and its
finite-domain limits are described in :doc:`menudet_accuracy`.

Release boundary
----------------

Portable DSL 1.0 and 1.1 reductions are named ``block_sum``, ``block_prod``,
``block_min``, ``block_max``, ``block_mean``, ``block_any`` and ``block_all``. Each takes one
operand and reduces only the supplied evaluation block's valid, participating
lanes. There is no ``axis`` or ``keepdims`` argument and no automatic combination
of results across blocks. Changing evaluation partitions can change results.
Bare reduction calls, including ``mean``, are rejected; existing draft artifacts
must be updated or re-exported. General lazy expressions and native array/graph
plans retain whole-array and axis reductions with their existing names.

``block_mean`` divides the ordered sum by the number of participating lanes.
Float32/float64 inputs retain their dtype; integer/Boolean inputs use a checked
integer sum and return float64. Integer accumulation overflow raises an error.
An empty or entirely masked block returns NaN.

``DSLKernel.export`` and ``validate_portable_dsl`` default to draft 1.0.
Native validation of normalized source, typed captures,
fixed widths and cardinality completes before export returns JSON::

    author = blosc2.DSLKernel.from_source("def total(x):\n    return block_sum(x)\n")
    record = author.export({"x": "int64"}, "int64",
                           cardinality="block_scalar")
    kernel = blosc2.PortableKernel.from_json(record)
    values = kernel.lazy({"x": np.arange(6, dtype="int64")}, partitions=(3,))[:]
    # array([3, 12])

The draft admits all standard Boolean/integer widths, float32/float64, fixed byte
and Unicode strings, mixed operand-driven numeric promotion, checked arithmetic,
native math/string operations, masked ordered block reductions and logical ND
coordinates. Specify ``ndim`` for coordinate-bearing sources. Existing static
``row["column"]``, NumPy calls and string syntax reuse the full-DSL authoring
normalizers; dynamic indexing, objects and callbacks are not reconstructed.

Raw draft source can also be checked without artifacts, arrays, execution or JIT::

    info = blosc2.validate_portable_dsl(
        "def coordinate(x):\n    return x + _i0\n",
        {"x": "int16"}, "int64", ndim=1,
        cardinality="elementwise")
    assert info["valid"]

The core native validator checks exact numeric/string signatures and reports
native status and source line/column. String dtypes specify exact slot widths,
for example ``{"x": "U4"}``; output widths must match native inference.
``cardinality=None`` checks inferred cardinality; an explicit declaration must
match. Reserved ND symbols require an explicit covering ``ndim``. This source
validator remains available when the optional artifact JSON loader is disabled.
It checks raw native syntax, not Python normalization, globals or runtime domains.

Supported features include elementwise arithmetic, comparisons and chains,
Boolean logic, local assignments, branches, bounded loops, and explicit output
conversions. Boolean operands participate in numeric arithmetic as zero/one;
truth conversion occurs at a Boolean output or explicit ``bool()`` call.

External functions, callbacks, object operations and unsupported dynamic indexing
are rejected during export and independently by native import; there is no Python fallback.

Integer overflow, out-of-range narrowing, non-finite/out-of-range float-to-integer
casts and integral zero divisors raise checked evaluation errors. The interpreter
establishes its floating-point environment and restores the caller's environment.
Callers must keep data and intermediates in the defined domains;
static validation cannot prove those conditions. This is not a sandbox for
untrusted source or arbitrary data. NaN payloads are not guaranteed.

Use ``blosc2.validate_portable_dsl(source, input_dtypes, output_dtype)`` to check
membership without execution or JIT. It returns validity, a status category,
and optional source diagnostics. Import/export errors raise
``blosc2.PortableArtifactError`` with a stable ``status`` attribute. Missing
returns, zero range steps, and loop-cap failures are runtime evaluation errors;
failed output contents are unspecified. Empty elementwise evaluations validate
bindings without execution; block-scalar reductions use their specified empty identities/errors.

Builds and execution
--------------------

Draft logical partitions and persistence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a validated 1.0 kernel, ``kernel.lazy(inputs, partitions=(...))`` binds an
immutable logical C-order block grid. This grid is independent of compressed
operand chunks/blocks. Elementwise output retains the original domain;
``block_scalar`` output has one item per logical partition (the grid shape).
Scalar returns under lane-varying branches/loops, or inside loops with
lane-varying break/continue, reject as ambiguous output—even if their scalar
expressions match. Uniform reduction conditions and coherent scalar control
flow remain supported; varying loops may rejoin before a top-level scalar return.
Slices evaluate intersecting **original complete groups** before selecting result
lanes. ND coordinates retain the original domain and group origin. Rechunking the
materialized result changes storage, never reduction membership::

    lazy = kernel.lazy({"x": stored_array}, partitions=(32, 64))
    values = lazy[1:3, ::2]
    result = lazy.compute(chunks=(8, 16), blocks=(4, 8))
    lazy.save("recipe.b2nd")
    reopened = blosc2.open("recipe.b2nd")

Native JSON records, signatures and logical partitions persist through disk
carriers, frames, structured MessagePack and EmbedStore/DictStore/TreeStore,
including arithmetic expressions containing portable carriers. Operands must
have persistent references; failures preflight before overwriting destinations.
Import never reconstructs a Python function. Safe arithmetic envelopes permit
only bound-name primitive arithmetic/Boolean expressions, not calls or attributes.

New DSLKernel-backed LazyUDF saves automatically normalize authoring and typed
captures, validate draft 1.0 natively and persist artifact/bindings/context. The
same preflight applies to frames, MessagePack, stores and nested arithmetic
references. Python UDFs and noncompliant normalized kernels reject before
destination writes; there is no source fallback. Positional scalar arguments
become typed constants, array arguments retain their parameter names, and the
current output shape and logical block partitions are preserved. A block-scalar
return cannot silently replace an elementwise LazyUDF shape: author it through
``export(cardinality="block_scalar")`` and ``PortableKernel.lazy``
with explicit partitions instead. Storage chunk boundaries must not split the
declared uniform logical block grid.

Historical source-backed files require explicit ``deserialize='full'``;
that policy propagates through nested references. Loaded legacy recipes are not
automatically migrated by saving. CTable legacy computed/generated DSL metadata
has the same explicit policy gate. The older table vector-column DSL API lacks
an explicit row-domain contract and cannot be silently converted to independent
rows when loading. New registrations use the independent-row contract below.
``msgpack_unpackb()`` also defaults to safe decoding. Active reconstruction needs
explicit ``deserialize='full'``; container readers propagate their effective
policy through nested values instead of relying on the helper's default.

Draft native table transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scalar stored columns bind to native transformers by name::

    table.add_computed_column("total", kernel, inputs={"x": "amount"})
    table.add_generated_column("stored_total", values=kernel,
                               inputs={"x": "amount"})

These APIs also accept a newly authored ``DSLKernel`` directly. Specify the
output dtype when it differs from input type promotion. Normalization, capture snapshots and native validation finish
before column registration or destination mutation::

    table.add_computed_column(
        "twice", blosc2.DSLKernel.from_source("def twice(x):\n    return x * 2\n"),
        inputs={"x": "amount"}, dtype="int64")

For a fixed-shape row reduction, additionally specify
``cardinality="block_scalar"``. Independent-row grouping is the contract of
kernel registration; there is no ``row_domain`` option or separate
``add_portable_*`` API. Positional input lists and column-bound DSL LazyUDFs
also adopt this contract on new registration. String/lazy expressions and
non-kernel row transformers retain their existing behavior.

The independent domain means **each scalar row is one original
one-lane group**, not a storage block or a changing table-wide reduction. Rank-one
coordinates are row-local: shape ``(1,)``, origin ``(0,)``, ``_i0 == 0`` and
``_n0 == 1``. Block-scalar and elementwise kernels both yield one scalar per row.
Named bindings may be empty for constant/ND constructors. Appending, refreshing,
deleting rows, materializing and compact-copying retain this mapping. Artifacts,
bindings and fixed row shapes survive table disk/frame round trips, with native
validation before binding and again before saving. Fixed-width numeric/string
signatures are immutable. Fixed-shape ndarray columns also support scalar native
row reductions: each complete row is one group, its broadcast row shape is the
logical ND domain, and its origin is zero. Scalar columns can broadcast within
that row without changing coordinates. Shape/cardinality incompatibilities reject
before registration. Multi-row table partitions and vector-valued row outputs
are not implicitly admitted. The draft remains subject to conformance and
external platform release gates.

The native artifact loader is enabled by default in Python package builds.
Custom builds may disable it with ``MINIEXPR_BUILD_ARTIFACT=OFF``; import then
raises ``NotImplementedError``. The optional adapter uses pinned, MIT-licensed
yyjson; the raw native compiler does not depend on it.

Draft 1.0 uses the typed interpreter. ``jit`` and compiler pragmas are preferences
only and currently fall back to interpretation; a required-JIT artifact rejects.
Semantic evaluation errors never retry the kernel. Artifacts
enforce strict FP independently of host defaults. Host while-loop limits remain
execution policy rather than artifact constants.

The authoritative language and artifact specifications are maintained in
miniexpr under ``doc/dsl-spec/1.0.md`` and ``doc/dsl-spec/artifact-1.0.md``.
Python-Blosc2 pins the published native revision that passed the native platform
matrix. Native CI is distinct from Python stock-wheel certification.

API
---

.. autoclass:: blosc2.PortableKernel
    :members:

.. autoclass:: blosc2.PortableArtifactError

.. autofunction:: blosc2.validate_portable_dsl
