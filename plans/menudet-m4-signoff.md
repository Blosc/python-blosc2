# M4 — functions and exceptional-value semantics

Date: 2026-10-09. Reference: NumPy **2.5.3**.

## Outcome and acceptance scope

M4's declared real function matrix and native floating-status policy are
implemented in opt-in portable **1.1**. Checked 1.0 exports, existing artifacts,
dependency pins and default backend routing are preserved. Implementation/matrix
acceptance is complete; cross-native-platform qualification remains pending
Linux/Windows CI execution, not silently inferred from macOS results.

The authoritative native corpus is
`../miniexpr/tests/numpy-compat/functions-v1.1.json`; its independent Python
generator/reference reader is `tools/menudet_functions.py`. The native contract
is documented in `../miniexpr/doc/numpy-functions-1.1.md` and Python usage in
`doc/reference/portable_dsl.rst`.

- **75 spellings, 825 signature rows, 1,154 passing contract cases.**
- **668 NumPy-reference cases**, **486 native-contract cases**, including
  **230 diagnostic cases** across the complete corpus. These categories must not
  be conflated into 1,154 NumPy numerical parity passes.
- **64 independently certified finite cases × 32 samples**, seed **20261009**,
  using 100-decimal-digit mpmath references. All discrete/exact and bounded
  transcendental comparisons have explicit policies. The 8-ULP sampled bound is
  not a proof of correctly-rounded libm or a global bound over all real inputs.
- Shared C/Python off/on execution agrees, with **empty NumPy reference drift**;
  provenance, corpus/runtime hashes and results are in
  `plans/menudet-m4-conformance.json`.
- Portable **JIT/SIMD eligibility remains zero**. On requests exercise the
  interpreter fallback, not accelerated semantic conformance.

## Implemented behavior

Classification, distinct `minimum`/`maximum` versus `fmin`/`fmax`, and aliases
`absolute`/`fabs` have native typed execution. The matrix includes arithmetic
helpers, rounding, sign, power, selection, logarithmic/exponential/trigonometric
and hyperbolic functions, stepping/sign primitives and existing libm extensions.
Tests cover all advertised real spellings and arities, accepted/rejected input
types, inferred dtype, normal/subnormal boundaries, signed zero, NaN/infinity,
domain limits, ties and large/small magnitudes.

The signature audit corrected `rint`, `fabs`, small-integer real-math promotion,
Boolean `square`/`conj`, Boolean `imag`, integer `fmin`/`fmax`/`fmod`, one-argument
ties-even `round`, divisor-sign `remainder`, and zero-sign `sign`. Float32 uses
float libm operations; no output dtype drives intermediate arithmetic.
NumPy float16-result signatures explicitly reject rather than silently widening.

Native `me_artifact_eval_status()` supplies cleared per-call stable invalid,
divide, overflow and underflow flags, configurable raise masks, caller fenv
restoration, scoped thread-local collection, valid-mask participation and
selected-lane branch semantics. Cross-block aggregation is caller-owned OR;
there is no mutable cumulative state on compiled handles. Python exposes
`return_status=True` and `fp_errors='ignore'|'raise'`; exceptions include
`.fp_status`. Default execution continues ignoring floating flags without warnings.

A full-suite regression exposed newly registered predicates entering legacy
full-DSL nullable-table routing. These spellings are now profile-gated to 1.1;
eight dedicated routing regressions and the existing 76-test nullable-logic module
verify that legacy routing remains unchanged.

## Explicit divergences / capability limits

- `where` evaluates only selected lanes; eager NumPy branch-argument diagnostics
  are intentionally not reproduced. Masked operations do not contribute flags.
- Extrema signed-zero ties follow deterministic IEEE min/max rules. NumPy tied
  zero bits vary with array length/SIMD/platform. Floating extrema vectors are
  marked native-contract rather than disguised NumPy references.
- Native extensions lacking a NumPy equivalent use explicitly declared contracts.
- Unsupported spellings, float16 loops, extra rounding arguments, ufunc methods
  and tuple-valued functions are recorded and reject source validation.
- Reporting is an IEEE-operation flag policy, not full `np.seterr`: no inexact,
  warning counts, callbacks, or synthesized integer floating warnings. Internal
  libm flags can differ from NumPy vector-loop diagnostics.
- WASM has no observable fenv exception flags: reports `supported=False`, not a
  false exception-free claim; raising rejects before execution. Values and policy
  rejection are tested; native flag assertions are capability-qualified.
- Linux/Windows execution, NumPy 1.26 integration, release-wheel qualification
  and eligible portable acceleration are outstanding qualifications.

## Reproduction

Use the `blosc2` conda environment. The editable extension must explicitly use
the sibling checkout; no dependency pin was updated:

```sh
conda run -n blosc2 python -m pip install -e . --no-build-isolation --no-deps \
  --config-settings=cmake.define.FETCHCONTENT_SOURCE_DIR_MINIEXPR=/Users/faltet/blosc/miniexpr
conda run -n blosc2 python -m tools.menudet_functions \
  ../miniexpr/tests/numpy-compat/functions-v1.1.json
conda run -n blosc2 python -m tools.menudet_conformance run \
  ../miniexpr/tests/numpy-compat/functions-v1.1.json \
  --native-runner "$BUILD/tests/numpy_compat_runner" \
  --work-dir "$WORK" --report plans/menudet-m4-conformance.json
MENUDET_NUMPY_FUNCTION_CORPUS=/Users/faltet/blosc/miniexpr/tests/numpy-compat/functions-v1.1.json \
MENUDET_NUMPY_RUNNER="$BUILD/tests/numpy_compat_runner" \
  conda run -n blosc2 pytest tests/test_menudet_functions.py -q -n 0
conda run -n blosc2 ctest --test-dir "$BUILD" --output-on-failure
```

Native CTest registers the same off/on function corpus for native platform CI,
sanitizers and Node/WASM. C and Python readers assert status behavior independently;
native reporting requests also check raise-mask errors and recovery. Python tests
add same-handle concurrent isolation and explicit cross-block aggregation.

## Final validation

- Full Python suite: **12,286 passed, 55 skipped**, after correcting the
  reproducible legacy nullable-table routing regression.
- Complete native CTest suite: **425 passed**.
- ASan/UBSan shared-corpus/example checks: **9 passed**.
- Standalone Node/WASM shared-corpus/example checks: **9 passed**.
- Focused Python M4 tests: **21 passed**, also **21 passed** against the
  sanitized C runner.
- Paired C/Python corpus: **1,154 cases × off/on**, all contract checks pass;
  reference drift is empty, eligible portable acceleration is zero.
- JSON Schema validation, Ruff lint/format and both repository whitespace checks
  pass; independent corpus regeneration is identical. Native, sanitizer and WASM
  rebuilds emitted no new compiler warnings.
- Sphinx HTML build succeeded; it emits existing project-wide missing autosummary,
  duplicate-label, theme and cross-reference warnings. No warning in the modified
  portable DSL reference was observed. Editable build configuration also emitted
  dependency CMake deprecation/policy warnings, not new compiler warnings.
