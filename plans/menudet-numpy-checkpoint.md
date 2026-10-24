# Menudet NumPy compatibility — first checkpoint results

2026-10-09. Implements section 13 of `menudet-numpy-compat.md`: inventory,
baseline evidence and a small shared conformance slice, **without arithmetic or
default-backend changes**. This records the original checkpoint; subsequent M1
sign-off and M2 completion are in `menudet-m1-m2-signoff.md`.

## Paired revisions and reference

| Component | Revision / configuration |
| --- | --- |
| Python branch start | `numpy-compat`, `063fd8b9a4c65ae75d76080f3497086a8e5a3e9d` |
| Native branch start | `numpy-compat`, `36770f87e10f1b92bfe0f9f92a7039eba2e976ea` |
| Committed native checkpoint/corpus | `4e45081889e20e6a3820822a808f92de00fff7e6` |
| Installed native dependency | established CMake pin `3418cdce4b5c11e1661c8b6453e3d31d94b0b2b4` |
| NumPy semantic reference | **2.5.3**, confirmed in `blosc2` conda environment |
| Verified host | macOS arm64, little endian, 64-bit intp, Python 3.14.4 |
| Reference policy | fixed explicit dtype widths, `np.errstate(all="ignore")`, exact bit comparisons |

Rebuilt the editable Python install with default FetchContent configuration in a
fresh build directory, `--no-build-isolation --no-deps`; verified the fetched native
checkout SHA above. No sibling source override or dependency-pin/default changes.
Native runtime sources at the pinned revision and the starting native SHA are
identical (the intervening commits change documentation only). The standalone
runner adds no numerical semantics. Exact extension SHA256/package path, runner
revision and local tooling changes are recorded in
`menudet-numpy-checkpoint-results.json`. This is paired starting-revision-plus-local-
changes evidence; the final Python change is identified by its containing revision.

Initial target set is Linux/macOS/Windows native 64-bit and WASM 32-bit; this
checkpoint verifies **only the host above**. NumPy 1.26 remains a host compatibility
target, not a second semantic reference. Generation refuses another NumPy version;
running reports secondary-version and value drift without rewriting golden data.

## What was implemented

Native repository owns the single reviewed corpus, schema and inventory:

- `tests/numpy-compat/vectors.json`: **33 vectors**, **14 matching / 19 divergent**
  explicit signatures. Both runners agree on status and successful output bits.
- `tests/numpy-compat/schema.json`: versioned, lossless big-endian bytes; dtype,
  shape, scalar category, weak literal type and IEEE signed-zero/exceptional bits.
- `tests/numpy-compat/inventory.json`: broader capability matrix, including
  unsupported and unverified types, functions, array operations and acceleration.
- `tests/numpy-compat/runner.c`: standalone artifact execution with explicit typed
  buffers, repeated execution/diagnostic checks and observed-baseline verification.
  CTest runs off/on requests and explicitly reports interpreter selection.
- `doc/numpy-compat-checkpoint.md`: traced dispatch/type/cast rules, divergence
  register (impact/cost/decision/evidence), platform defaults and transition policy.

Python integration adds:

- `tools/menudet_numpy_compat.py`: pinned reference generation, installed-extension
  execution, cross-runner recording, baseline verification, drift detection and
  preliminary performance measurements. No source `eval`, sibling-library loading
  or independently maintained NumPy arithmetic rules.
- `tests/test_menudet_numpy_compat.py`: bit roundtrips, version guard, shared corpus,
  authoring export/typed captures, dtype/shape preservation, NumExpr-call prohibition,
  future artifact/language rejection and malformed native-vector regression tests.
- Array inventory probes demonstrate rejection of singleton broadcasting, a scalar
  block sum of 15 versus NumPy axis sums [3,12], negative-stride copying-adapter
  value agreement and unsupported half/complex dtypes.

The corpus covers all signed/unsigned integer addition boundaries, multiply,
negate/abs minima, exact >2**53 integers, weak float32 literals, typed captures,
0-D arrays, signed/unsigned promotion/comparison, Boolean addition, true/floor
division and remainder, zero/min divisors, shifts, finite/nonfinite casts,
checked output narrowing and floating signed zero. The nonfinite NumPy integer
sentinel is labelled platform-qualified rather than made a universal contract.

Known NumPy failures are **classified evidence**, not xfailed tests or arithmetic
fixes. Baseline tests pass by preserving those observations. Matching output
signatures are not proof of inferred/intermediate dtype parity. General tolerance,
ULP policies, reference diagnostics, layout recipes, fuzzing/minimization and
WASM/other-platform vector adapters remain milestone-2 work.

## Verification

- Native interpreter-only full CTest: **327 passed**.
- Native corpus with JIT off/on request: both passed, actual backend interpreter.
- ASan + UBSan corpus runner: both off/on requests passed, no reported findings.
- Python default suite after a fresh pinned build: **12,220 passed, 55 skipped**.
  No new shared-corpus tests were skipped when explicit
  corpus/runner paths were provided. Existing platform/capability skips remain.
- Final focused checkpoint tests: **16 passed**, including malformed vectors
  executed against the ASan/UBSan runner and the four array/dtype inventory probes.
- Ruff checks and formatting pass. Native builds introduced no reported warnings.

Reproduction from this repository, selecting native artifacts explicitly:

```sh
export MENUDET_NUMPY_CORPUS=/absolute/path/to/miniexpr/tests/numpy-compat/vectors.json
export MENUDET_NUMPY_RUNNER=/absolute/path/to/native-build/tests/numpy_compat_runner
conda run -n blosc2 pytest tests/test_menudet_numpy_compat.py -q -n 0
conda run -n blosc2 python tools/menudet_numpy_compat.py run "$MENUDET_NUMPY_CORPUS" \
  --native-runner "$MENUDET_NUMPY_RUNNER" \
  --native-revision 4e45081889e20e6a3820822a808f92de00fff7e6 \
  --installed-native-revision 3418cdce4b5c11e1661c8b6453e3d31d94b0b2b4
```

`generate` and `record` deliberately rewrite reviewed vectors; ordinary `run`
never does. `benchmark OUTPUT.json` writes separate performance evidence.

## Next vertical slice and semantic/artifact handling

Select **fixed-width add/subtract/multiply, negate and abs**, all integer widths.
Eleven of the nineteen divergent vectors directly motivate this narrowly scoped
slice. Native checked helpers localize the work; implement wrapping through
well-defined unsigned/bit operations, then audit accelerated paths independently.
Division/zero diagnostics, shifts, scalar promotion and casts follow separately.

Before incompatible arithmetic, require **language 1.1 + artifact schema 1.1** and
a distinct native semantic profile. Do not silently reinterpret checked draft 1.0
artifacts. Retain old interpretation during transition or explicitly reject old
artifacts; opt-in re-export is migration. Current 1.1 rejection is tested, but the
new profile/version is not implemented or advertised by this checkpoint. This
policy does not require permanent competing arithmetic personalities.

## Feasibility of the remaining plan

| Work | Assessment |
| --- | --- |
| Finish conformance machinery | High feasibility; extend the existing shared format/runners incrementally, including inferred-type and diagnostic assertions. Native-host adapters and cross-platform reference qualifications need explicit work. |
| Integer arithmetic/promotion/casts | High feasibility, moderate cost. Checked helpers and typed nodes provide a strong base. Boolean rules, weak float context, signed/unsigned 64-bit promotion and cast policies need operation-specific changes, not just wrapping. |
| Functions and exceptional values | Feasible but substantial verification effort. FP-status aggregation, masked branches, float32 intermediates, platform libm differences and JIT agreement are more difficult than adding function names. |
| Native broadcasting/reductions | Feasible, high architectural cost. Existing block execution is not native NumPy array traversal. Shape/stride ownership, empty dimensions, grouping, masks and accumulator defaults must be designed behind a C API. |
| Graph integration / acceleration | Highest near-term risk. The typed interpreter is semantically useful but currently far too slow for NumPy/NumExpr-class throughput. Eligible JIT/SIMD/fusion and reusable plans require semantic-preserving optimization, not relabelling fallback. |
| 1.0 freeze/release | A useful declared subset is feasible. Full NumPy, complex/half extensions, general indexing and view/aliasing parity should remain outside the initial promise unless measured value justifies their cost. |

`menudet-numpy-checkpoint-benchmark.json` records an affine expression, float32/64
and int64, 16 and 262,144 elements, compilation/first/steady timings, NumPy and
NumExpr controls. At the larger size the current integrated interpreter takes
roughly **0.49 seconds**, versus sub-millisecond controls. These are preliminary,
single-process fixed-order timings with allocated outputs and 12 NumExpr threads;
they are not controlled release budgets or universal slowdown ratios. The native
runner also records separate CPU microtimings. Larger direct-native timing, RSS,
thread scaling, graph scheduling and alternating subprocess controls remain work.

Conclusion: proceed with the integer/revision slice; numerical compatibility is
tractable, but performance and native array orchestration must be explicit go/no-go
checkpoints. Do not infer default-switch readiness from semantic baseline success.
