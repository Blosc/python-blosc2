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

``DSLKernel.export`` and ``validate_portable_dsl`` default to draft 1.0.
Native validation of normalized source, typed captures,
fixed widths and cardinality completes before export returns JSON::

    author = blosc2.DSLKernel.from_source("def total(x):\n    return sum(x)\n")
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
