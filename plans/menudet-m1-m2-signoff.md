# Menudet M1 sign-off and M2 completion

2026-10-09. M1 specification/inventory/baseline and M2 conformance-harness
acceptance are complete for the verified **macOS arm64** host. This does not
claim NumPy arithmetic compatibility or cross-platform qualification. Runtime
arithmetic, artifact/language 1.0, dependency pins and default backends are unchanged.

## M1 remaining sign-off items

- Actual TCC and Homebrew **GCC 14.4.0**, not Apple's gcc/Clang alias, are verified
  by compiled program handles. Fallback is rejected by the benchmark executable.
- Three subprocess trials, alternating backend order, cover float32/float64/int64,
  16 and 262,144 elements. Cold compile, first execution, warm median,
  same-process recompile and fresh-process GCC disk reuse are distinct fields.
  Tiny native warm calls are batched to avoid timer-resolution artifacts.
- Owned per-trial caches and a compiler-invocation wrapper verify one GCC compile
  for cold runs and zero for disk reuse. TCC's in-memory route has no persistent
  disk-cache claim. Same-process recompilation is timing evidence, not an
  instrumented TCC cache-hit claim.
- Direct-native portable versus installed Python-integrated portable baselines
  locate the dominant cost in the portable interpreter profile, not Python copying.
- Inventory review replaces broad function/layout uncertainty with bounded shared
  evidence. Broad full-interpreter/full-JIT dtype parity remains deliberately
  `unverified`; performance controls cannot resolve that claim.

Median float64 large-array warm times (`menudet-m1-backend-baselines.json`):

| Route | Time |
| --- | ---: |
| Direct portable interpreter | 483 ms |
| Python-integrated portable interpreter | 478 ms |
| Full-DSL interpreter control | 0.509 ms |
| Full-DSL TCC control | 0.392 ms |
| Full-DSL GCC control | 0.175 ms |
| NumPy | 0.143 ms |
| NumExpr, one thread | 0.228 ms |

Large float64 cold compilation medians: TCC 1.06 ms; GCC 386 ms. GCC fresh-process
disk-load compilation is 0.28–1.11 ms with zero compiler invocations. These are
single-host affine positive-domain baselines, not release performance budgets.
Native output is preallocated; Python output allocation is included. No RSS,
thread-scaling, complex-domain or general semantic equivalence claim is made.

## M2 delivered

- Native-owned v2 schema/corpus and independent C/Python readers, preserving v1.
- **64 shared cases: 41 matching, 21 known divergences, 2 named capability skips**.
  Both off/on requests agree in dtype, shape, diagnostics, comparisons and outcome.
- Lossless fixed-width transport, weak/typed/0-D scalar categories, physical layout
  recipes, NaN payload cases, exact/bitwise/ULP/tolerance comparison policies.
- Separate reviewed divergence and observed baseline checks: new failures are not
  silently converted into passing golden data. Reports identify every mismatch,
  missing capability, and actual backend; portable acceleration eligibility is zero.
- Seeded PCG64 properties and deterministic lane/scalar minimization, with the
  overflow failure promoted into a stable shared vector.
- Repeated results/diagnostics, floating-environment restoration, failure-to-success
  recovery using the same compiled handle, and malformed-vector regression tests.
- Standalone `tests/numpy-compat/example.c` artifact/typed-buffer consumer.
- Independent **284-sample** finite mpmath certification retained and passing.

Evidence: `menudet-m2-conformance.json`, `menudet-m2-properties.json`,
`menudet-m2-math-certification.json`; native format contract is documented in
`miniexpr/doc/numpy-compat-conformance.md`. Runner/extension hashes and revisions
are recorded; native numerical sources still match the established installed pin.

## Reproduction

Validation completed: **12,248 Python tests passed, 55 skipped**; **421 native
CTest tests passed** with TCC enabled. The 28 focused M2 tests also pass against
the ASan/UBSan runner, including malformed-vector probes; all four checkpoint/v2
off/on sanitizer CTest checks pass. The v2 corpus validates against its JSON
Schema. Ruff formatting/lint and whitespace checks pass. Other platforms remain
unverified.

From Python-Blosc2, select the paths explicitly (using your own checkout/build):

```sh
conda run -n blosc2 python -m tools.menudet_conformance run "$CORPUS_V2" \
  --native-runner "$RUNNER" --work-dir "$OWNED_WORK_DIR" --report report.json
conda run -n blosc2 python -m tools.menudet_conformance properties "$CORPUS_V2" \
  --native-runner "$RUNNER" --work-dir "$OWNED_WORK_DIR" --report properties.json
MENUDET_NUMPY_CORPUS_V2="$CORPUS_V2" MENUDET_NUMPY_RUNNER="$RUNNER" \
  conda run -n blosc2 pytest tests/test_menudet_conformance.py
conda run -n blosc2 python scripts/certify_menudet_math.py
```

Generation requires NumPy 2.5.3 and explicit `generate --checkpoint "$CORPUS_V1"`.
Recording is a separate reviewed action; normal runs never rewrite corpus data.
JSON Schema validation is development tooling only, not a native runtime dependency.

Next is M3's explicit version-transition prerequisite and fixed-width arithmetic
slice. Platform qualification and eligible portable acceleration stay visible as
future work rather than being disguised as passing interpreter tests.
