# M3 — arithmetic, promotion and conversion

Date: 2026-10-09. Reference: NumPy 2.5.3, macOS arm64.

## Outcome and acceptance qualification

M3 implementation is complete for the **declared** bool, signed/unsigned
8/16/32/64-bit and float32/64 dtype/operator/cast matrix. The opt-in schema and
language **1.1 / 1.1** profile is independent of checked 1.0; no saved artifacts
are upgraded or silently reinterpreted. Source validation, native execution,
metadata-only inference, Python lowering, captures and persistence are covered.

The plan's broader **cross-native-platform acceptance is not yet signed off**:
Linux and Windows jobs register the corpus, but have not been executed here.
Verified execution is macOS arm64 and standalone Node/WASM. No eligible portable
JIT/SIMD implementation exists; explicit on requests execute the interpreter,
and required acceleration rejects. Full-DSL compiler controls are not evidence
of portable arithmetic conformance. There is no new accelerated path to qualify.
NumPy 1.26 integration and release wheels also remain unqualified.

## Implemented slices

- Fixed-width modular add/subtract/multiply/power, unary negation/absolute value,
  signed minima, integral true/floor division and remainder, bitwise operations,
  all comparisons and defined negative/oversized shifts. Signed arithmetic uses
  unsigned bits and explicit reconstruction instead of undefined C overflow.
- Boolean restrictions/result types, strong signed/unsigned/float promotion,
  weak literals, weak plain numeric captures, strong NumPy scalar captures and
  typed zero-dimensional inputs. Exact mixed-integer comparisons intentionally
  differ from the float64 arithmetic promotion of int64/uint64 pairs.
- Metadata-only inference before output conversion; public native
  `me_artifact_inferred_dtype()` and Python `PortableKernel.inferred_dtype`.
  Changing output width does not change intermediate arithmetic width.
- Native fixed-width cast syntax, modular array integer narrowing, checked weak
  scalar construction, direct float-width conversion and Boolean truth casts.
  `safe` / `same_kind` / `unsafe` output policies validate before reading data.
- NumPy's corrected floating floor division; float32 intermediates stay float32,
  multiply/add do not contract, and native evaluation restores the caller's FP
  environment. WASM tests numerical values under its fixed-nearest environment;
  alternate rounding and IEEE exception-flag probes are host-only.
- Artifact scalar categories, authoring captures and persisted lazy recipes
  preserve version/profile/type intent without reconstructing Python functions.

Defined divergence: floating-to-integer conversion truncates but rejects
nonfinite or truncated out-of-range values, including negative unsigned results,
rather than preserving platform/compiler-dependent NumPy sentinel integers.
Weak-only integers are bounded by int64 transport; arbitrary-precision Python
arithmetic is not implemented. Aggregated NumPy-like floating warnings/status
remain M4 work. Array broadcasting is outside this matrix.

## Corpus and evidence

Native-owned `../miniexpr/tests/numpy-compat/arithmetic-v1.1.json` contains
**3,193 cases**: **2,793 independently regenerable NumPy value references** and
**400 native-contract diagnostic cases**. Contract diagnostic successes must not
be described as 400 extra NumPy numerical parity passes. The corpus includes all
121 strong dtype pairs, the operator/type matrix, 363 policy/type combinations,
boundary casts, weak scalar comparisons/construction, signed-minimum/-1,
zero divisors, extreme shifts, captures and noncontraction controls.

`tools/menudet_arithmetic.py` generates the corpus from pinned NumPy, with an
independent declarative ufunc recipe for drift checking (no source eval).
`plans/menudet-m3-conformance.json` records paired standalone/native and integrated
Python off/on results, hashes and provenance: every case matches its declared
contract, reference drift is empty and eligible JIT cases are zero. Inferred
dtype checks are independent of requested output dtype in both readers.

Seeded PCG64 property tests use seeds 1729 and 20261009, each with 128 cases of
16 full-width random lanes across all eight integer widths. They cover wrapping,
floor division, remainder, extreme shifts and comparisons rather than only safe
small-value addition. Historical M1/M2 checked-profile vectors remain unchanged.

## Reproduction

Validation results:

- Full Python suite: **12,265 passed, 55 skipped**.
- Full native CTest suite: **423 passed**.
- ASan/UBSan checkpoint/conformance/arithmetic/example checks: **7 passed**.
- Focused Python integration tests against the sanitized C runner: **17 passed**.
- Node/WASM checkpoint/conformance/arithmetic/example checks: **7 passed**.
- JSON Schema validation, Ruff lint/format and both repositories' whitespace
  checks pass. Native/extension builds introduce no compiler warnings.

One initial full-suite invocation used a nonexistent v2 corpus filename; those
setup errors disappeared after correcting the selector to `vectors-v2.json`.
An initial sanitizer invocation had not built the standalone example target;
building it and rerunning all seven checks passed. Neither was a code regression.
The first WASM configuration emitted CMake's shared-library fallback warning;
explicitly selecting `MINIEXPR_BUILD_SHARED=OFF` removed it before validation.

All Python, build and test commands use the `blosc2` conda environment. The editable
development extension was built with the **explicit** modified native checkout;
the repository dependency pin and default backends were not changed:

```sh
conda run -n blosc2 python -m pip install -e . --no-build-isolation --no-deps \
  --config-settings=cmake.define.FETCHCONTENT_SOURCE_DIR_MINIEXPR=/Users/faltet/blosc/miniexpr

conda run -n blosc2 python -m tools.menudet_arithmetic \
  /Users/faltet/blosc/miniexpr/tests/numpy-compat/arithmetic-v1.1.json

conda run -n blosc2 python -m tools.menudet_conformance run \
  /Users/faltet/blosc/miniexpr/tests/numpy-compat/arithmetic-v1.1.json \
  --native-runner "$NATIVE_BUILD/tests/numpy_compat_runner" \
  --work-dir "$WORK_DIR" --report plans/menudet-m3-conformance.json

MENUDET_NUMPY_ARITHMETIC_CORPUS=/Users/faltet/blosc/miniexpr/tests/numpy-compat/arithmetic-v1.1.json \
MENUDET_NUMPY_CORPUS=/Users/faltet/blosc/miniexpr/tests/numpy-compat/vectors.json \
MENUDET_NUMPY_CORPUS_V2=/Users/faltet/blosc/miniexpr/tests/numpy-compat/vectors-v2.json \
MENUDET_NUMPY_RUNNER="$NATIVE_BUILD/tests/numpy_compat_runner" \
MINIEXPR_ARTIFACT_RUNNER="$NATIVE_BUILD/tests/portable_artifact_runner" \
conda run -n blosc2 pytest -q

conda run -n blosc2 ctest --test-dir "$NATIVE_BUILD" --output-on-failure -j 8
conda run -n blosc2 ctest --test-dir "$SANITIZED_BUILD" -R numpy_compat --output-on-failure
conda run -n blosc2 ctest --test-dir "$WASM_BUILD" -R numpy_compat --output-on-failure
```

Build roots under the approved OpenCode temporary directory:

- Native: `menudet-m1-jit-build` (existing TCC/GCC-capable controls).
- Sanitized: `menudet-checkpoint-sanitize` (ASan/UBSan).
- Python extension: `menudet-m3-python-build`.
- WASM: `menudet-m3-wasm-build`, Emscripten toolchain, Node emulator, artifact
  support on, shared library/TCC/WASM JIT/Accelerate/SLEEF off. Runner uses
  NODERAWFS, a 1 MiB stack and explicit exit-runtime handling. The original small
  WASM stack exhausted on the larger corpus; this is a runner resource setting,
  not an arithmetic exception bypass.

The native contract is documented in `../miniexpr/doc/numpy-arithmetic-1.1.md`;
Python usage is in `doc/reference/portable_dsl.rst`.
