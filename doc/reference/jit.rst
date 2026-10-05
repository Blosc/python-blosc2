.. _JITOptions:

JIT execution options
=====================

Blosc2 can compile eligible computations at runtime (JIT compilation). Native
builds use the bundled Tiny C Compiler (TCC) by default, so installing a system
compiler is **not** necessary to use Blosc2 or its default JIT.

Global defaults and scoped options
----------------------------------

Use :func:`blosc2.set_jit_options` to configure a script, or
:func:`blosc2.jit_options` for a temporary thread/task-local override:

.. code-block:: python

    import blosc2

    previous = blosc2.set_jit_options(jit_backend="cc")
    a = blosc2.linspace(0, 10, 10_000)
    expr = a * 2 + 1
    with blosc2.jit_options(jit=True, trace=True, cflags="-O2"):
        result = expr.compute()
    blosc2.set_jit_options(**previous)

Both functions accept the same keyword-only parameters:

.. list-table::
    :header-rows: 1
    :widths: 25 20 55

    * - Parameter
      - Built-in default
      - Meaning
    * - ``jit``
      - ``None``
      - Existing best-effort JIT policy; ``True`` also requests eligible plain
        expression auto-lifting. ``False`` disables JIT.
    * - ``jit_backend``
      - ``None``
      - ``"tcc"``, ``"cc"``, or ``"js"``. JavaScript is WebAssembly/Pyodide-only.
    * - ``fp_accuracy``
      - ``FPAccuracy.DEFAULT``
      - Existing floating-point function accuracy policy, not a compiler flag.
        Use an actual :class:`blosc2.FPAccuracy` member.
    * - ``trace``
      - ``False``
      - Python routing and native compilation/cache/fallback diagnostics on
        stderr. Does not enable external compiler output.
    * - ``compiler``
      - ``None``
      - CC compiler command, normally ``cc``. A string or path-like object.
        Quoted commands can name executables with spaces; path-like objects
        are converted to shell-quoted executable paths automatically.
    * - ``cflags``
      - ``None``
      - Additional CC flags as a string, appended to the normal optimization and
        floating-point flags. Compiler-specific; changing them can affect accuracy.
    * - ``cache_dir``
      - ``None``
      - Actual persistent CC cache directory, not a ``TMPDIR`` root. Accepts a
        string/path-like object and resolves relative paths when configured.
        Created on demand only by CC; TCC does not use it.
    * - ``compiler_output``
      - ``False``
      - Opt into the external compiler's output independently of ``trace``.

Compiler commands/flags are trusted shell configuration, not safe inputs from
untrusted users. Choose protected cache storage. CC-specific options have no
effect on TCC, JavaScript, or non-JIT execution.

For setters and contexts, an omitted argument leaves/inherits that setting;
``None`` resets it to the built-in default. The setter returns the previous
process defaults as a fresh dictionary. :func:`blosc2.get_jit_options` returns
a fresh dictionary of the current Python defaults, including context overrides,
but excluding environment overrides and per-evaluation arguments. Validation is
atomic: an invalid setting leaves the defaults unchanged.

Within Python, precedence is:

**explicit evaluation settings → explicit LazyUDF settings → context defaults →
process defaults → built-in defaults**.

The execution options also work as per-call kwargs on the evaluation entry points
listed below. For these existing per-call APIs, ``None`` means inherit, not reset;
use a resetting context to select built-in defaults locally. ``fp_accuracy`` now
defaults to ``None`` on lazy compute/reduction methods, so omitted accuracy
inherits rather than masking configured defaults. Without configured defaults,
behavior is unchanged.

Settings resolve at evaluation time: an expression constructed outside a context
can be evaluated inside it. Explicit settings attached to a LazyUDF remain
authoritative unless overridden on its ``compute()`` call. Changes do not modify
already compiled kernels. Contexts nest and restore settings even after exceptions;
they never mutate environment variables or process-wide defaults. Async tasks
inherit their creation context but do not affect unrelated tasks. Thread context
propagation follows Python's ``contextvars`` rules (including explicit copied
contexts, ``asyncio.to_thread()``, and Python's thread-context inheritance setting).
Process-wide setters still affect all threads without a contextual override.

Compiler command/flags and explicit cache directories participate in native cache
identity, including process-local positive/negative caches. Changing them must not
reuse an incompatible kernel. Tracing/compiler-output toggles do not change cache
identity. ``fp_accuracy`` retains the existing evaluator accuracy semantics;
it does not select miniexpr's separate strict/contract/fast compiler mode.

Existing nonempty environment overrides remain authoritative: ``CC`` overrides
``compiler``, ``CFLAGS`` overrides ``cflags``, ``ME_DSL_TRACE`` overrides ``trace``,
``ME_DSL_JIT_DEBUG_CC`` overrides ``compiler_output``, and
``ME_DSL_JIT_CACHE_DIR`` overrides ``cache_dir``. ``TMPDIR`` is used only when no
explicit cache directory is selected. The JIT enable/backend environment rules
below remain unchanged. Configuration does not require changing environment
variables; native compilation receives a call-local settings snapshot.

.. autofunction:: blosc2.set_jit_options

.. autofunction:: blosc2.get_jit_options

.. autofunction:: blosc2.jit_options

JIT is best effort
------------------

``jit=True`` prefers JIT; it does not require successful native compilation.
If TCC or the explicitly selected system compiler cannot compile or load a
kernel, miniexpr evaluates it with its interpreter instead. This also applies
when SELinux or another operating-system policy denies executable mappings.
Fallback is quiet by default and may be slower. Invalid arguments, invalid DSL
syntax, unsupported backend names, and genuine computation errors still raise
their normal errors.

Successful array creation alone is not evidence that JIT ran. Use
``ME_DSL_TRACE=1`` to inspect runtime compilation, cache reuse, and fallback, or
:func:`blosc2.validate_dsl_jit` to probe a DSL kernel for specific input/output
dtypes. The latter probes compilation, not execution on real data.

Per-call controls
-----------------

These options are accepted by :meth:`blosc2.LazyArray.compute` and
:func:`blosc2.lazyudf`. :func:`blosc2.arange` and :func:`blosc2.linspace` also
accept them through ``**kwargs``. Eligible lazy-expression reductions, such as
``expr.sum(jit=True, jit_backend="cc")``, forward them to evaluation. These
settings tune execution rather than array storage; they are not options to
:func:`blosc2.empty`.

``jit=None`` (default)
    Use the entry point's default policy. DSL kernels, including the real-valued
    ramps used by ``arange`` and ``linspace``, try JIT. Plain expressions do not
    automatically become DSL kernels unless JIT is requested. Ordinary Python
    callback UDFs are not compiled by TCC/CC through this option.

``jit=True``
    Prefer JIT for eligible computations. Plain expressions can be automatically
    lifted into DSL kernels. Interpreter fallback remains available.

``jit=False``
    Disable JIT for the computation. Applicable environment overrides are
    described below. Depending on the expression, the non-JIT route can be the
    miniexpr interpreter or another existing compute engine.

``jit_backend=None`` (default)
    Use the default backend: TCC on supported native targets. Under
    WebAssembly/Pyodide, eligible floating-point DSL kernels prefer the JavaScript
    backend unless ``jit=False`` or ``strict_miniexpr=True`` is specified.

``jit_backend="tcc"``
    Select bundled TCC. It compiles generated C in memory, retains the compiled
    state for evaluation, and does not persist binaries for another Python
    process. It creates no JIT source, binary, metadata, or cache-directory files.
    On Linux, executable storage uses anonymous memfd-backed RW/RX mappings;
    it does not need a writable/executable temporary directory. Installed
    libraries still need to be loadable normally. If the target lacks a TCC
    backend, or allocation is denied, execution uses the interpreter.

``jit_backend="cc"``
    Select an installed C compiler (normally GCC or Clang), with optimized code
    generation. This needs a compiler and writable cache storage that permits
    loading executable shared libraries. Compiled libraries and metadata persist
    for reuse by subsequent processes. Compiler absence, compilation failure,
    cache failure, or denied library loading falls back to the interpreter.
    For a plain expression, also set ``jit=True`` to request DSL auto-lifting.

``jit_backend="js"``
    Select the JavaScript bridge, available only under WebAssembly/Pyodide.
    Explicit selection on native builds raises an error. Eligibility and explicit
    unsupported-kernel errors are described in the
    `DSL syntax reference <dsl_syntax.html#execution-backends>`_.

``fp_accuracy`` controls numerical accuracy independently of the backend.
TCC currently supports the strict miniexpr compiler floating-point mode;
unsupported JIT modes can use the interpreter. Backends need not produce
bit-identical floating-point results. ``strict_miniexpr`` controls whether
miniexpr compilation/evaluation failures are raised instead of changing compute
engines; it is **not** a requirement that native JIT succeed. A valid kernel may
still use miniexpr's interpreter.

Examples
--------

.. code-block:: python

    import blosc2

    a = blosc2.linspace(0, 10, 10_000)  # Default: try bundled TCC
    a = blosc2.linspace(0, 10, 10_000, jit=False)
    a = blosc2.arange(10_000, jit=True, jit_backend="tcc")
    a = blosc2.linspace(0, 10, 10_000, jit=True, jit_backend="cc")

    expr = a * 2 + 1
    result = expr.compute(jit=True, jit_backend="cc")  # NDArray


    @blosc2.dsl_kernel
    def squared(x):
        return x * x


    udf = blosc2.lazyudf(squared, (a,), dtype=a.dtype, jit_backend="tcc")
    values = udf[:]  # NumPy values

Environment settings and precedence
-----------------------------------

Set environment variables before starting Python. Python-level and miniexpr-level
controls have different scopes; neither is an indication that JIT actually ran.

.. list-table::
    :header-rows: 1
    :widths: 32 68

    * - Variable
      - Meaning
    * - ``BLOSC_ME_JIT``
      - On ``LazyExpr.compute()`` (and paths calling it), values ``1``, ``true``,
        or ``on`` override ``jit`` to ``True`` without changing a supplied backend.
        Values ``tcc`` and ``cc`` additionally override ``jit_backend``.
        This is not a universal constructor/``LazyUDF`` override. ``0`` does not
        disable JIT: use per-call ``jit=False`` or ``ME_DSL_JIT=0`` instead.
    * - ``ME_DSL_JIT=0``
      - Disable miniexpr runtime JIT, including when a Python call requests
        ``jit=True``. This does not control the separate JavaScript backend.
    * - ``ME_DSL_TRACE=1``
      - Print native code-generation/runtime diagnostics to stderr, including
        builds, cache hits, failure explanations, and interpreter fallback.
    * - ``BLOSC_ME_JIT_TRACE=1``
      - Print Python compute-engine routing on supported paths to stdout.
        Distinct from native JIT success reporting.
    * - ``ME_DSL_JIT_DEBUG_CC=1``
      - Show the external compiler's normally suppressed output.
    * - ``CC`` and ``CFLAGS``
      - Select and configure the system compiler backend; defaults to ``cc``
        with optimized compilation. Not needed by TCC.
    * - ``TMPDIR``
      - CC cache root: artifacts go into ``$TMPDIR/miniexpr-jit``. When unset,
        Linux/macOS use ``/tmp/miniexpr-jit-<uid>``. TCC does not use this cache.
    * - ``ME_DSL_JIT_CACHE_DIR``
      - Exact CC cache directory; overrides Python ``cache_dir`` and the
        ``TMPDIR``-based default. TCC ignores it.
    * - ``ME_DSL_JIT_TCC_OPTIONS``
      - Extra options for compiling generated C with TCC. Not build flags that
        change libtcc's executable allocator.
    * - ``ME_DSL_JIT_LIBTCC_PATH``
      - Advanced: explicitly override the loaded libtcc path. A failed override
        does not silently substitute another library; evaluation can fall back.
    * - ``ME_DSL_JIT_POS_CACHE=0``
      - Disable applicable process-local positive-cache reuse. This does not
        disable persistent CC disk-cache reuse.
    * - ``ME_DSL_JIT_COMPILER``
      - Advanced miniexpr-wide compiler override, ``tcc`` or ``cc``. It overrides
        native compiler selection in DSL pragmas, including the selection emitted
        by Python's ``jit_backend`` option. Prefer per-call options ordinarily.

Tracing and restricted environments
-----------------------------------

.. code-block:: sh

    env ME_DSL_TRACE=1 python -c \
      'import blosc2; blosc2.linspace(0, 10, 10_000, jit=True, jit_backend="tcc")'

Representative runtime messages include:

.. code-block:: text

    [me-dsl] jit runtime built: fp=strict compiler=tcc key=...
    [me-dsl] jit runtime hit: fp=strict source=disk-cache key=...
    [me-dsl] jit runtime fallback: interpreter fp=strict compiler=tcc reason=...

An executable-memfd denial can disable TCC JIT. A ``noexec`` cache mount or an
SELinux file-execution restriction can disable CC JIT. Neither normally prevents
a valid computation from running with the interpreter. No system-policy changes
are required to use that fallback.

A fresh Python process is not necessarily a cache-cold CC run: it can reuse an
earlier shared library. Use a fresh ``TMPDIR`` when measuring CC compilation.
Measure Python startup, compilation/linking, first library loading, and array
execution separately when comparing backends. TCC prioritizes startup latency;
CC often improves throughput on large computations. Writing a persistent user
array with ``urlpath=`` is independent of JIT artifact storage.
