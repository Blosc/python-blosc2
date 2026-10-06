Portable DSL artifacts
======================

Portable profile 0.1 is a conservative, versioned subset of the full native DSL.
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
and matching shapes. Supported dtypes are ``bool``, ``int32``, ``int64``,
``float32``, and ``float64``; all parameters and captures share one dtype, while
output may differ. Captured supported Python/NumPy scalars are snapshotted into
typed constants, never live callbacks or arrays. Integers use range-checked
decimal strings; floats use exact IEEE-754 hexadecimal bits.

Release boundary
----------------

Supported features include elementwise arithmetic, comparisons and chains,
Boolean logic, local assignments, branches, bounded loops, and explicit output
conversions. Boolean operands participate in numeric arithmetic as zero/one;
truth conversion occurs at a Boolean output or explicit ``bool()`` call.

Excluded features include transcendental calls (including ``sin``/``cos``),
nested ``int()``/``float()`` casts, expression arguments to casts in floating-input
signatures, mixed-input dtypes, uncertified mixed computation intermediates,
integer division, float32 arithmetic inside predicates, reductions, indexing,
floating-spelled literals in integral-input signatures,
shape symbols, and external functions. A separately assigned float32 arithmetic
local can be compared after rounding. Excluded source is rejected both during
export and independently by native import; there is no Python fallback.

Integer overflow, out-of-range narrowing, non-finite/out-of-range float-to-integer
casts, zero divisors, and non-default floating rounding modes are outside the
runtime contract. Callers must keep data and intermediates in the defined domains;
static validation cannot prove those conditions. This is not a sandbox for
untrusted source or arbitrary data. NaN payloads are not guaranteed.

Use ``blosc2.validate_portable_dsl(source, input_dtypes, output_dtype)`` to check
membership without execution or JIT. It returns validity, a status category,
and optional source diagnostics. Import/export errors raise
``blosc2.PortableArtifactError`` with a stable ``status`` attribute. Missing
returns, zero range steps, and loop-cap failures are runtime evaluation errors;
failed output contents are unspecified. Empty evaluations validate bindings but
do not execute the kernel.

Builds and execution
--------------------

The native artifact loader is enabled by default in Python package builds.
Custom builds may disable it with ``MINIEXPR_BUILD_ARTIFACT=OFF``; import then
raises ``NotImplementedError``. The optional adapter uses pinned, MIT-licensed
yyjson; the raw native compiler does not depend on it.

The interpreter is the baseline. TCC/CC are optional accelerators, selected with
``jit`` and native compiler pragmas. Missing accelerators can fall back to native
interpretation. Semantic evaluation errors never retry the kernel. Artifacts
enforce strict FP independently of host defaults. Host while-loop limits remain
execution policy rather than artifact constants.

The authoritative language and artifact specifications are maintained in the
pinned miniexpr dependency under ``doc/dsl-spec/0.1.md`` and
``doc/dsl-spec/artifact-0.1.md``. The feature boundary is frozen; certification
on additional release platforms is tracked by CI. Broader numeric features and
lazy/container integration are deferred to a later profile.

API
---

.. autoclass:: blosc2.PortableKernel
    :members:

.. autoclass:: blosc2.PortableArtifactError

.. autofunction:: blosc2.validate_portable_dsl
