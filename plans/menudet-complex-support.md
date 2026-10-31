# Menudet complex support: implementation and handoff plan

## 1. Goal and scope

Implement a consistent **interpreter-first complex contract** for Menudet on
macOS, Linux, and WASM32. Support `complex64` and `complex128` inputs, outputs,
constants, captures, locals, and intermediate values. Windows must reject
complex kernels explicitly, while continuing to build and execute real kernels.

Complex evaluation must not call the originating Python function, use NumPy as
an execution fallback, or retry the full/legacy DSL. NumPy is a numerical test
oracle, not a runtime dependency of a native artifact.

Complex JIT is **not required** for the first implementation. `jit=True` is a
best-effort request: a kernel requiring complex computation should execute in
the typed interpreter if its JIT backend cannot implement it. Select that path
before execution; never replay a partially executed failing JIT kernel.

“Transparent fallback” means transparent JIT-to-interpreter selection within
the supported Menudet contract. It does not mean accepting every NumPy
operation or silently executing unsupported language constructs.

Out of scope initially:

- Windows complex execution, including MinGW-specific exceptions.
- Complex JIT/AOT bridges, vectorized complex math, and performance tuning.
- float16, extended precision, records, datetime/timedelta, objects, or
  variable-length string support.
- Changing trusted legacy recipe semantics or removing their explicit opt-in.

## 2. Working environment and state

Two repositories are involved:

- Python-Blosc2: `/Users/faltet/blosc/python-blosc2`
- Native miniexpr: `/Users/faltet/blosc/miniexpr`

Read both repositories' `AGENTS.md` before editing. Use the `blosc2` conda
environment for Python, builds, and tests. Do not assume the editable install
loads the extension just rebuilt; print both module paths before testing.

This plan was written after the initial migration of source-authored LazyUDFs
to Menudet. There are also **uncommitted native and Python changes** extending
artifact-load options and compatibility dispatch. Inspect `git status` and
diffs in both repositories; preserve this work. Do not reset the repositories.
Do not assume the dependency pin includes those uncommitted changes.

An earlier investigation mistakenly compared the pinned miniexpr build with
source from a newer sibling checkout. Avoid repeating that error: record the
actual source override, native revision/dirty status, build directory, and
loaded extension for every result.

Committing/pushing requires the user's explicit instruction. Follow repository
rules for CI monitoring, particularly `gh run watch --interval 60 --exit-status`.

## 3. Verified baseline: do not overstate legacy support

The following observations motivated this work:

1. On native macOS, forced legacy execution of identity and `z*z + 1` succeeds
   for both complex widths. Successful execution with `jit=True` does **not**
   establish that a complex JIT kernel was generated.
2. Native Windows compilation is explicitly rejected in `src/miniexpr.c` with
   `ME_COMPILE_ERR_COMPLEX_UNSUPPORTED` and a C99-complex-ABI diagnostic.
3. Python-Blosc2 disables complex legacy DSL kernels on Windows **and WASM** in
   `src/blosc2/lazyexpr.py`. Constructors and ordinary expressions can use
   separate NumPy paths; those are not evidence of native DSL support.
4. After rebuilding the local WASM target, Node execution of
   `build-wasm-menudet/tests/test_complex_functions.js` passed all 10 tests:
   conjugation, real/imaginary extraction, and scalar/complex division.
5. Additional standalone WASM complex128 DSL probes passed arithmetic, local
   assignment, a loop, and complex truth-value branching. They reported
   `jit=0`, confirming interpreter execution.
6. Legacy's `is_complex_supported_function()` in `src/functions.c` admits a
   limited whitelist: basic arithmetic, negation, powers, square root,
   absolute value, conjugation, real/imaginary extraction, and reduction nodes.
   Comparison nodes are rejected there. Admitting reduction nodes is not a
   claim that all reductions or numerical edge cases are qualified.
7. Legacy generated JIT code excludes complex dtypes (`src/dsl_jit_cgen.c`).

Conclusion: reuse appropriate numerical primitives, not legacy's overall
portability or correctness claims. Windows requires more foundational work;
WASM already has useful C complex interpreter machinery.

## 4. Code map to audit

### miniexpr

- `src/miniexpr.h`: dtype IDs, public values/views, compile/evaluation errors.
- `src/miniexpr.c`: Windows rejection, compilation admission, option scoping.
- `src/functions.c`, `src/functions.h`: existing complex operations and platform
  adapters; identify which helpers are safe to reuse.
- `src/dsl_compile.c`, `src/dsl_compile_support.c`: locals, type inference,
  expressions, literals, return conversion, control-flow joins.
- `src/dsl_portable_types.{c,h}`: promotion, casts, scalar numerical rules.
- `src/dsl_portable_expr.c`: typed admission and evaluation, truth values,
  comparisons, unary/binary operations, batch-path eligibility.
- `src/dsl_portable_reduce.h`: reduction behavior and accumulator types.
- `src/dsl_eval.c`: frame/local storage, copying values, masks, JIT dispatch.
- `src/dsl_artifact.c`, `src/miniexpr_artifact.h`: manifest types, constants,
  capabilities, widths, descriptors, overlap/alignment checks, load options.
- `src/dsl_array.c`: array-view widths, byte order, broadcasting, strides,
  reductions, storage readers, output writers.
- `src/dsl_graph*.c`: graph typing, serialization, scheduling and source views.
- `src/dsl_portable_jit.c` and JIT backend/cache files: early ineligibility,
  trace reasons, source generation and cache invalidation if needed.
- `tests/test_complex_functions.c`, `tests/test_dsl_portable_types.c`,
  `tests/test_dsl_artifact.c`, `tests/test_portable_jit_parity.c`, and array/graph
  tests: existing rejection assumptions and reusable test infrastructure.
- Native language/artifact specifications and `.github/workflows/ci.yml`.

### Python-Blosc2

- `src/blosc2/portable_kernel.py`: dtype/scalar descriptors, weak captures,
  manifest export, validation, loading, block/array execution APIs.
- `src/blosc2/dsl_kernel.py`: source normalization, captures, dtype inference.
- `src/blosc2/blosc2_ext.pyx`: NumPy dtype mapping, buffer widths/alignment,
  descriptors, native error translation, prefilter admission and scratch sizing.
- `src/blosc2/dsl_artifact_bridge.h`: optional-runtime capability shims.
- `src/blosc2/dsl_execution.py`: migrated LazyUDF dispatch, chunk-local groups,
  scalar-return broadcasting, dtype gates and native JIT requirements.
- `src/blosc2/proxy.py`: `@blosc2.jit` control-flow wrapper.
- `src/blosc2/{storage_readers,resident_execution,native_graph}.py`: execution
  admission, operation safety assumptions, size calculations and fallback paths.
- `src/blosc2/ndarray.py`, `src/blosc2/lazyexpr.py`: constructor/expression
  compatibility policies. Do not remove all legacy guards in one sweep.
- `doc/reference/portable_dsl.rst` and the portable/DSL/expression tests.

Use targeted searches: the above is an audit map, not a requirement to modify
every file. Existing complex dtype IDs in Cython do not imply portable support.

## 5. Phase A: settle the contract before implementation

Write a short native specification change and tests for the decisions below.
Use the project's declared NumPy-style numerical profile as the reference,
not assumptions about C complex behavior.

### Representation and manifest

- Define exact array storage: two components of the declared float width,
  component order, byte order, size, and alignment.
- Define endian-stable constant/capture encoding preserving each component's
  signed zero, infinities, and NaN bits where the existing contract requires it.
  Prefer an extension of the existing IEEE-hex encoding rather than JSON numbers
  or Python `repr(complex)`; document the precise byte layout.
- Define whether source `j` literals are admitted. Reuse the existing grammar
  where sound; exporting a complex capture must not depend on literal syntax.
- Distinguish ordinary Python complex captures (weak scalars) from NumPy complex
  scalars (strong typed scalars), including explicit capture dtype overrides.
- Decide capability/version signaling. Do not silently reinterpret an old
  constant encoding or assume an older 1.0 reader accepts newly added dtypes.
  Prefer capability detection/rejection consistent with the current draft policy.

### Numerical behavior

Specify and test:

- Strong and weak promotion for bool, all supported integers, float32/64, and
  complex64/128; mixed branch/local joins; inferred real outputs from complex
  inputs. Output dtype must not determine intermediate precision.
- Truth tests, equality/inequality, logical operations, `where`, and selection.
- Ordering, extrema, sorting-related semantics: **measure the declared NumPy
  reference behavior**. Do not assume NumPy rejects all complex comparisons;
  either implement the chosen contract or reject it explicitly in the first cut.
- Unsupported arithmetic such as floor division, remainder, and bitwise
  operations: reject during validation where statically identifiable.
- Cast policies `safe`, `same_kind`, and `unsafe`; real-to-complex and
  complex-to-real/integer behavior. Decide imaginary-discard diagnostics without
  assuming Python warnings are a portable native error mechanism.
- Arithmetic precision, division by zero, overflow/underflow, nonfinite values,
  signed zeros, and floating-point flags/support reporting.

Avoid “every kernel works” claims until an operation support matrix exists.

## 6. Phase B: native dtype foundation and artifact transport

1. Admit complex64/128 in portable type validation and public dtype descriptors
   only on supported platforms. Preserve the real-only Windows build.
2. Audit every width helper and scalar union/frame slot. Complex128 is 16 bytes;
   several current batch/scratch assumptions are bounded by `sizeof(double)`.
3. Add exact scalar load/store, copying, captures, serialization and cast support.
4. Extend artifact constant validation, itemsize/byte-count checks, overflow
   protection, alignment and overlap checks.
5. Audit array/reader byte-order conversion: swap the two components separately,
   not the whole 16-byte value. Cover strided, broadcast and zero-dimensional
   input views.
6. On Windows, reject a complex input, constant, local, or intermediate even
   when the final result is real. A real output dtype is not sufficient admission.

Implementation cautions:

- Keep public headers buildable with Windows compilers. Do not add unconditional
  C99 `_Complex` requirements to portable public data structures.
- Verify native C complex layout against the chosen buffer ABI; use safe copies
  or component conversion instead of relying on aliasing-unsafe pointer casts.
- Construct complex infinities/signed zeros carefully. `real + I*imag` can have
  different exceptional-value behavior from a proper component constructor.
- Audit ABI/cache versioning before changing shared scalar layouts or tables.

Acceptance: native and Python-exported artifacts round-trip complex buffers and
constants on the three supported platforms; Windows rejects them clearly.

## 7. Phase C: interpreter arithmetic and control flow

Implement the typed scalar evaluator first:

1. Identity, unary negation, addition, subtraction, multiplication, division,
   conjugation and real/imaginary extraction.
2. Complex locals, reassignment, branch joins, truth tests, equality and `where`.
3. Lane-local `if`, `for`, `while`, `break` and `continue` using the existing
   control-flow contract. Never substitute whole-array Python truth semantics.
4. Masks and initialization tracking. Inactive/uninitialized lanes must not be
   evaluated or contribute errors/FP flags.
5. Real-valued outputs such as `abs(z)`, `real(z)`, and `imag(z)` with the correct
   component/result precision and output casting policy.

Initially exclude complex expressions from the bounded real/integer batch
engine. Correct scalar interpretation is preferable to a real fast path that
silently drops imaginary components or overwrites scratch storage.

Acceptance: identical supported results with JIT requested and disabled, no
Python callback, no full-DSL fallback, deterministic unsupported-operation errors.

## 8. Phase D: reductions and complex math coverage

### Reductions

- Implement/qualify sum, product, mean, logical reductions, and accumulator dtype
  rules before advertising general reduction support.
- Define variance/std outputs and conjugation rules if those operators belong
  to the current language. Defer unsupported operators explicitly.
- Cover empty groups, masks, `initial` where supported, NaNs, infinities, and
  scalar cardinality. Preserve complete logical groups across slicing/rechunking.
- Decide extrema/comparison semantics from Phase A, not legacy's whitelist.

### Transcendentals

Add a declared operation set incrementally: square root and power first, then
exp/log families, trigonometric, inverse and hyperbolic functions as supported
by the language. Audit each existing primitive against the reference profile.

Test branch cuts from both sides using positive and negative zero, extreme
magnitudes, mixed real/complex inputs, and complex64 rounding. C99 libm behavior
can differ from NumPy and across host libraries; wrappers or narrow documented
exclusions may be necessary. Do not advertise bitwise transcendental parity
across platforms without evidence.

## 9. Phase E: JIT policy, Python dispatch and storage integration

1. Reject complex JIT admission based on the **entire typed program**, including
   inputs/intermediates feeding a real output. Check generic, arithmetic and ramp
   bridges; trace a useful reason for interpreter selection.
2. Propagate Windows unsupported errors rather than disguising them as a JIT
   miss or falling back to NumPy/legacy.
3. Extend Python scalar/dtype export and inference, preserving weak/strong
   captures. Add native capability checks rather than just deleting `dtype.kind`
   gates.
4. Enable supported complex `LazyUDF` and native `@blosc2.jit` control-flow calls
   through Menudet. Preserve per-call settings, output identity, casts, slicing,
   persistence and original chunk-local reduction groups.
5. Extend portable block/array/reader execution and only then eligible compressed
   prefilters. Recheck scratch widths, padded edge scattering and FP handling for
   16-byte output. A general portable path may remain the first correct route.
6. For WASM, test portable interpretation directly before changing Python's
   dispatch guards. Native compression prefilter scheduling remains disabled on
   WASM; that is independent of complex expression evaluation.
7. Keep explicit JS backend behavior separate. Do not present the existing JS
   transpiler as a complex-capable Menudet artifact interpreter. If `js` cannot
   execute complex kernels, define explicit interpreter selection or a clear
   unsupported error; do not silently use legacy native DSL.
8. Remove only obsolete guards with corresponding platform tests. Trusted saved
   legacy recipes must keep their own compatibility policy.

## 10. Testing strategy and completion criteria

### Ownership

Put exhaustive dtype/operator/promotion/edge-value matrices in **miniexpr**.
Python-Blosc2 tests should cover bindings, dispatch, buffer/storage contracts,
and a representative end-to-end set, not duplicate every native cross-product.

### Native coverage

- Both complex widths, weak/strong scalars, mixed real/integer types, casts.
- Constants and captures, including malformed manifests, byte order, signed
  zeros, infinities, NaNs, capacity/alignment/overlap failures and empty buffers.
- Arithmetic, control flow, masks, reductions and the supported math matrix.
- Real outputs from complex inputs and complex intermediates from real inputs.
- JIT-off/JIT-requested parity; interpreter selection reported accurately.
- Windows compile/load rejection without breaking real kernels or headers.
- macOS/Linux interpreter tests and WASM Node tests using **rebuilt** artifacts.
- ASan/UBSan checks where supported, especially new widths and scalar frames.

### Python coverage

- `DSLKernel.export` -> `PortableKernel.from_json` -> block and array evaluation.
- Complex64/128 ndarray, NDArray, scalar and captured inputs.
- `lazyudf` and `@blosc2.jit` control-flow execution, including real outputs.
- Monkeypatch legacy hooks/Python function bodies to fail, proving the migrated
  path actually uses Menudet; account for intentionally unsupported platforms.
- NumPy/NDArray outputs, partial and strided selections, padded 2-D/3-D blocks,
  persistence and reload, empty arrays, and non-native-endian inputs.
- Compiler/cache/trace options and failure isolation remain correct.
- Explicit Windows/JS errors and WASM interpreter behavior.

### Qualification

Do not call the feature complete until there is an exact-revision matrix for
macOS, Linux and WASM covering the declared contract, plus Windows negative
coverage and a green existing real/string suite. Record known numerical
differences and unsupported operations explicitly in the specifications/docs.

## 11. Build and verification examples

Run from Python-Blosc2, using the local native override during iteration:

```bash
conda run -n blosc2 python -m pip install -e . --no-build-isolation \
  --config-settings=build-dir=build-portable-python \
  --config-settings=cmake.define.FETCHCONTENT_SOURCE_DIR_MINIEXPR=/Users/faltet/blosc/miniexpr
conda run -n blosc2 python -c 'import blosc2; print(blosc2.__file__); print(blosc2.blosc2_ext.__file__)'
conda run -n blosc2 python -m pytest -n 0 tests/test_dsl_execution.py
conda run -n blosc2 python -m pytest --tb=short
```

Prefer incremental native builds when configured correctly; pip build isolation
can leave cached NumPy include paths pointing at deleted temporary directories.
Do not overwrite another checkout's installed extension during isolated tests.

From miniexpr, discover the relevant test targets before using these examples:

```bash
conda run -n blosc2 cmake --build build-portable --parallel 8
conda run -n blosc2 ctest --test-dir build-portable --output-on-failure
conda run -n blosc2 cmake --build build-wasm-menudet --target test_complex_functions --parallel 8
node build-wasm-menudet/tests/test_complex_functions.js
```

Existing WASM complex tests validate legacy primitives, not the new Menudet
contract. Add and run portable-specific targets too. Do not interpret a passing
suite with skipped Windows tests as evidence of Windows complex support.

## 12. Suggested delivery sequence

1. Contract/specification and failing native tests for representation/promotion.
2. Native artifact transport and basic typed interpreter arithmetic.
3. Control flow, masks, casts and real-valued complex operations.
4. Reduction contract and qualified math operations in bounded increments.
5. Python bindings and native LazyUDF/@jit integration.
6. WASM portable interpreter integration and Windows rejection qualification.
7. Documentation, platform support matrix, and representative benchmarks.

Keep each step independently reviewable. Benchmark setup separately from
evaluation and check actual JIT status. Performance is secondary to a sound
portable contract; do not add a complex AOT bridge until correctness is qualified.
