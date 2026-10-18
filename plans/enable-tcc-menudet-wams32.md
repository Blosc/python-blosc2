# Enable TCC JIT for Menudet on wasm32

Status: bridge-free wasm32 first slice implemented and qualified under Emscripten
Node (main and side modules), headless Chromium (main and side modules), and a
fresh Python-Blosc2 Pyodide wheel through the public API. Performance remains
workload- and target-dependent; typed bridges remain deferred.

## Objective

Enable the existing TCC-to-WebAssembly runtime to execute a deliberately narrow,
semantically exact subset of Menudet portable 1.1 kernels under Emscripten and
Pyodide. Start with bridge-free elementwise kernels, including exact `int64` and
`uint64` values held in WebAssembly memory. Unsupported kernels must continue
through the interpreter without changing results, status, masking, diagnostics,
or the public best-effort meaning of `jit=True`.

This is not a request to enable all host JIT lowering on wasm32. In particular,
JavaScript `Number` must never transport an exact 64-bit integer, and host function
pointers stored in the generated kernel's `inputs[]` table must not be assumed to
have the same ABI in a separately instantiated wasm module.

## Starting point and blocker

Paths prefixed with `miniexpr/` and `minicc/` refer to sibling repositories,
available locally as `../miniexpr` and `../minicc` during development.

| Area | Main files |
| --- | --- |
| Menudet typed C lowering and wasm exclusion | `miniexpr/src/dsl_portable_jit.c` |
| wasm32 TCC compilation, source patching, instantiation and cache | `miniexpr/src/dsl_jit_backend_wasm32.c` |
| JIT IR and runtime dispatch | `miniexpr/src/dsl_jit_ir.*`, `miniexpr/src/dsl_jit_runtime_nonhost.c` |
| Bridge definitions and program state | `miniexpr/src/dsl_jit_bridge_contract.h`, `miniexpr/src/dsl_compile_internal.h` |
| Browser/side-module helper | `miniexpr/src/me_jit_glue.js`, `miniexpr/tests/test_dsl_jit_side_module.c` |
| wasm native tests | `miniexpr/tests/test_dsl_jit_side_module.c`, `miniexpr/tests/test_me_jit_glue_smoke.js` |
| Python wasm integration | `CMakeLists.txt`, `src/blosc2/wasm_jit.py`, `tests/test_menudet_jit.py` |
| Pyodide build workflow | `.skills/wasm32-pyodide-dev/SKILL.md` |

`pj_prepare_source()` currently rejects every Emscripten target. That guard is
necessary because the existing wasm adapter patches legacy generated source by
replacing `int64_t`/`uint64_t` declarations and casts with 32-bit types. Menudet's
generated source requires true 64-bit values for typed array elements, modular
integer operations, constants, checked arithmetic state, ND metadata, and some
bridge arguments. Removing the guard while retaining that rewrite would silently
miscompile kernels.

The exported kernel itself can retain a wasm32-compatible JavaScript signature:

```c
int me_dsl_jit_kernel(const void *const *inputs, void *output, int count);
```

All three values cross JavaScript as wasm `i32`; typed values remain in wasm linear
memory. Internal generated code may use wasm `i64`, `f32`, and `f64`. The first
slice must not make an `i64` cross the JavaScript boundary or an indirect callback
boundary.

## Required behavior

| Situation | Expected behavior |
| --- | --- |
| Explicit portable 1.1 `jit=True`, eligible bridge-free kernel | Compile with TCC, instantiate wasm, and execute the generated kernel |
| `int64`/`uint64` input or output | Preserve all 64 bits in wasm memory, including values outside JavaScript's exact-number range |
| Participating mask | Skip inactive lanes before loads, arithmetic, and stores |
| Unsupported operation, bridge, reduction, rank feature, or ABI | Do not generate/compile a wasm Menudet kernel; use the interpreter |
| TCC compilation or wasm validation/instantiation failure | Clean up and use the interpreter; warnings remain opt-in |
| Missing side-module helpers | Use the interpreter, as today |
| `jit=False` | Do not attempt JIT |
| Native host target | Preserve current TCC/CC lowering and bridge behavior |

No fallback may turn a valid operation into an exception merely because JIT is
unavailable. Existing expression validation errors and runtime semantic errors
must remain errors rather than being hidden as JIT fallback.

## First-slice eligibility boundary

The initial wasm32 slice is intentionally bridge-free. It should admit only
operations whose complete behavior is emitted into the generated module:

- elementwise loads and stores for Boolean, signed/unsigned integer, float32 and
  float64 arrays;
- same-width modular integer `+`, `-`, `*`, unary `-`, and audited bitwise forms
  already lowered without callbacks;
- float32/float64 `+`, `-`, `*`, and `/`, subject to the existing portable
  exception/status contract that wasm32 can actually observe;
- same-domain integer comparisons and ordinary finite floating comparisons only
  if lowering proves they require no comparison bridge;
- Boolean operations, lazy `where`, branches, and lane-local bounded loops when
  every contained expression remains bridge-free;
- participating masks and simple uniform captures whose values are read from wasm
  memory without checked conversion or JavaScript transport.

Initially reject:

- checked-operation, integer-operation, comparison, predicate, and math callbacks;
- every generated kernel with nonzero `portable_jit_nchecked`,
  `portable_jit_niops`, `portable_jit_ncomparisons`, `portable_jit_npreds`,
  `portable_jit_nmath`, `portable_jit_nmath2`, or `portable_jit_nmath3`;
- reductions and scalar-output recipes;
- ND reserved symbols and ND metadata until their exact `int64_t` layout has a
  dedicated wasm test;
- local scratch arrays transported through nested pointer tables;
- weak/scalar conversions that can report overflow or otherwise need a bridge;
- NaN-sensitive comparisons and math operations whose interpreter parity depends
  on a callback or floating-status behavior not observable on wasm32;
- accelerated SLEEF/Accelerate math policies and relaxed floating-point modes.

Eligibility must be based on typed lowering state, not source-text heuristics. A
kernel that becomes ineligible after partial lowering must release its generated
source/IR and leave the program in the normal interpreted state.

## Task A: separate Menudet source from legacy wasm source rewriting

### A1. Identify generated-source capabilities explicitly

- [x] Add a generated-source kind or capability field to the JIT program/IR, for
  example legacy wasm-patchable versus portable typed wasm. Do not infer this from
  comments or search for type names in C source.
- [x] Set the capability in `dsl_portable_jit.c` only after portable lowering and
  wasm eligibility checks succeed.
- [x] Include the source kind, portable lowering revision, semantic profile and
  wasm ABI revision in the runtime cache identity. A legacy patched module must
  never alias a typed Menudet module.
- [x] Preserve native source behavior: native TCC and CC continue to consume the
  existing portable C source without wasm-specific substitutions.

### A2. Split backend preparation paths

- [x] Refactor `dsl_wasm32_patch_source()` into an explicitly legacy-only path.
  Keep existing substitutions unchanged for already-supported legacy kernels.
- [x] Add a typed portable wasm preparation path that does **not** replace
  `int64_t`, `uint64_t`, their casts, constants, or helper signatures.
- [x] Avoid broad textual rewriting in the new path. Generate the required wasm
  entry signature and preamble from `dsl_portable_jit.c`, or apply narrowly scoped,
  marker-backed substitutions whose inputs are tested byte-for-byte.
- [x] Compile with `-nostdlib -nostdinc` as today, while ensuring the generated
  portable preamble supplies everything it uses. Audit `stdint.h`, `limits.h`,
  `string.h`, `memcpy`, compound literals, unions, `UINT64_C`, and `restrict`
  against MiniCC's wasm frontend.
- [x] Add generated compile-time width checks for 8/16/32/64-bit integer typedefs
  where MiniCC supports them. Compilation failure must remain recoverable.

## Task B: define a wasm32-safe kernel ABI

### B1. Keep boundary values wasm32-sized

- [x] Give the exported wasm kernel an explicit count type that lowers to `i32`.
  Validate negative counts before converting to an internal nonnegative lane count.
- [x] Keep input-table pointers, input pointers, output pointers, masks, and status
  pointers as wasm32 addresses. Check assumptions about `uintptr_t` and pointer size.
- [x] Use an internal `int64_t` only where Menudet semantics require it. Prefer a
  checked `uint32_t`/`size_t` lane index for memory addressing in the first slice,
  so pointer arithmetic cannot wrap through a signed 32-bit expression.
- [x] Validate `count * itemsize` and every derived address range before execution,
  using the existing caller bounds where available. Do not rely on a wasm trap as
  ordinary bounds validation.
- [x] Retain `addFunction(fn, "iiii")` for the first-slice exported function and
  document why no `i64` is exposed to JavaScript.

### B2. Keep exact values inside wasm

- [x] Confirm MiniCC emits wasm `i64` loads, stores and arithmetic when compiling
  exact-width C types without the legacy rewrite.
- [x] Ensure signed conversion helpers preserve the same two's-complement modular
  bit patterns as native portable lowering. Do not use JavaScript `Number`, JSON,
  or double casts as an intermediate representation.
- [x] Verify alignment behavior for 64-bit typed loads/stores. Either prove callers
  supply the required alignment or generate wasm-safe unaligned accesses/copies.
- [x] Keep constants such as `INT64_MIN`, `UINT64_MAX`, shift counts and literal
  suffixes intact through MiniCC parsing and wasm emission.

## Task C: implement fail-closed wasm eligibility

- [x] Move the unconditional `__EMSCRIPTEN__` rejection in `pj_prepare_source()`
  behind a dedicated wasm eligibility decision rather than deleting it outright.
- [x] Check the target after typed lowering has populated bridge counters, but
  before publishing `jit_c_source` and its fingerprint.
- [x] Require elementwise output, supported rank policy, supported dtypes, no ND
  metadata, no scratch pointer table, and zero bridge counters for the first slice.
- [x] Audit operations that emit apparently local helper code but still depend on
  host status, floating environment, or interpreter callbacks. Add explicit flags
  where bridge counters alone are insufficient.
- [x] Emit one concise trace reason for each broad rejection category: bridge ABI,
  reduction/scalar output, ND context, scratch locals, unsupported semantics, or
  backend/compiler failure.
- [x] Ensure failed specialization with runtime constants cannot leave stale bridge
  counters, source, IR, fingerprint, or cache state on the program.
- [x] Keep `jit=True` best effort. Do not add a Python-visible strict-JIT mode as
  part of this work.

## Task D: update wasm compilation and instantiation

- [x] Dispatch legacy and portable source kinds separately in
  `dsl_jit_compile_wasm32()`.
- [x] Register no host symbols for a first-slice portable kernel except unavoidable
  runtime primitives whose signatures are explicitly audited. Prefer generated
  local helpers over imported `memcpy`/`memset` for small fixed-width operations.
- [x] After TCC output, validate the module's imports and exported kernel signature
  before instantiation. Reject unexpected function, memory, table, or global imports.
- [x] Preserve shared linear-memory import patching and remove any exported module
  memory exactly as the current adapter requires.
- [x] Keep module compilation and instantiation errors recoverable and warning-free
  by default. Opt-in traces should identify source kind and failing phase.
- [x] Replace the fixed `/tmp/me_jit_kernel.wasm` name with a collision-safe or
  in-memory mechanism before enabling concurrent portable compilation. If MiniCC
  cannot emit to memory, create a unique virtual-FS path and remove it on every
  success and failure path.
- [x] Audit cache ownership: cached function-table entries retain required scratch
  memory, eviction removes the JS function, and failed modules retain neither.
- [x] Include the portable wasm ABI revision in positive and negative cache keys.

## Task E: native wasm tests for the first slice

### E1. Generated-source and eligibility tests

Eligibility is checked through the artifact-level differential harness in both
Emscripten configurations (and the browser target) rather than a separate
host-only predicate copy; that keeps the assertions on the production lowering.

- [x] Cover eligibility acceptance/rejection through the artifact-level wasm
  differential harness, including generated-source width checks and the exported
  signature, rather than a separate copy of the predicate.
- [x] Assert portable wasm source retains `int64_t`/`uint64_t` and 64-bit exact
  values through the boundary corpus; the preamble asserts fixed widths/pointer
  size at compile time.
- [x] Assert legacy source still receives its existing compatibility patch and
  does not use the portable path (existing legacy wasm regression).
- [x] Assert math, predicate, checked-cast, mixed-comparison and reduction
  families each reject the first-slice wasm path and preserve values.
- [x] Assert native portable JIT source and host tests are unchanged.

### E2. Actual Emscripten execution

- [x] Extend `test_dsl_jit_side_module.c` or add a focused portable wasm test that
  requires `me_expr_has_jit_kernel()` for eligible cases, rather than merely
  accepting interpreter parity.
- [x] Test `int64` values around `0`, `-1`, `INT32_MIN/MAX`, `2^53`, `2^63-1`, and
  `INT64_MIN`; test `uint64` values around `2^32`, `2^53`, `2^63`, and `UINT64_MAX`.
- [x] Test passthrough, addition, subtraction, multiplication, unary negation,
  comparisons and selected bitwise operations, including modular wraparound.
- [x] Test float32 and float64 arithmetic with finite normal, subnormal, signed-zero,
  infinity and NaN inputs where the operation is eligible. Compare bits where the
  semantic contract requires bits and classify unavoidable wasm FP differences.
- [x] Test all-active, mixed and all-inactive masks. Inactive lanes must not read
  invalid input or trigger an otherwise participating semantic error.
- [x] Test lazy `where`, `if`/`elif`/`else`, and bounded loops with eligible bodies.
- [x] Test zero length, one lane, odd lengths, tile boundaries and the largest lane
  count practical in CI.
- [x] Assert the contiguous `me_artifact_eval_ex` fast ABI rejects naturally
  unaligned pointers; Python/array-view paths normalize or gather such views
  before execution instead of passing them to typed wasm loads.
- [x] For every rejected category, assert `has_jit == false` and interpreter values
  and errors remain correct.
- [x] Run both main-module and side-module helper configurations. Unregistering
  side-module helpers restores clean interpreter fallback.
- [x] Run repeated compile/free/cache-eviction cycles; this wasm build is
  single-threaded, so concurrent compilation is not applicable.

## Task F: Python-Blosc2/Pyodide integration

- [x] Update the local miniexpr source override first; change Python-Blosc2's
  bundled revision only after the miniexpr implementation and tests are stable.
- [x] Add focused Python tests using the public `@blosc2.menudet`/portable artifact
  path with `jit=True` and `jit=False`. Require actual JIT only on a known-capable
  wasm build; retain ordinary fallback tests elsewhere.
- [x] Cover NumPy `int64` and `uint64` arrays containing values above `2^53` so a
  JavaScript-number transport regression cannot pass accidentally.
- [x] Exercise C-contiguous arrays, masks, scalar captures accepted by the first
  slice, branches and loops through real Python/Cython dispatch.
- [x] Confirm unsupported reductions, math calls, checked conversions and ND
  features use the interpreter and preserve Python-visible diagnostics.
- [x] Build and reinstall a fresh Pyodide wheel using the repository wasm skill.
  Verify that both `blosc2` and its extension come from that wheel/runtime.
- [x] Run focused tests first, then the available wasm test suite with deterministic
  single-thread settings. Record Pyodide, Emscripten, browser/Node and MiniCC
  revisions in qualification results.

## Task G: typed bridge ABI follow-up

Do not begin this phase until bridge-free exact-i64 execution is qualified. The
preferred design is direct wasm imports with declared signatures, not function
pointers recovered from `inputs[]` and called through C casts.

- [ ] Define a versioned bridge manifest shared by C and JavaScript. Each import
  records module/name and exact wasm signature.
- [ ] Introduce bridges incrementally, beginning with one operation whose complete
  semantic behavior is easy to test. Candidate signatures include:
  - checked operation: `i64 (i32 node, i64 a, i64 b, i64 c, i32 status)`;
  - float64 unary math: `f64 (i32 node, f64 x)`;
  - comparison: `i32 (i32 node, f64 x, f64 y)`.
- [ ] Never pass an imported/exported `i64` through JavaScript without WebAssembly
  BigInt integration and explicit runtime support. Prefer wasm-to-wasm table/import
  calls or memory-based request/response records where portability requires it.
- [ ] Validate import signatures before instantiation and reject modules with an
  ABI revision mismatch.
- [ ] Preserve lazy participation and immediate status checks: a failing left
  operand must prevent evaluation of inactive right operands.
- [ ] Add one bridge family at a time with interpreter differential tests, edge
  vectors, masks and fallback tests. Do not enable all existing counters together.
- [ ] Address reductions, ND context and scratch locals only after their memory ABI
  is independently specified and tested.

## Benchmark and decision gate

Correctness qualification precedes performance claims. Once the first slice is
stable, add a wasm-specific benchmark that records separately:

1. typed source generation;
2. TCC compilation and wasm file emission;
3. `WebAssembly.Module` compilation;
4. `WebAssembly.Instance` creation and function-table registration;
5. first kernel execution;
6. warmed execution and cache hits;
7. interpreter execution for the same workload.

Use arithmetic, branch, loop and masked workloads at small, medium and large lane
counts. Include one-shot and repeated calls, report the reuse break-even count,
and compare values before reporting timings. Record Node/browser, Pyodide,
Emscripten, MiniCC, CPU and memory settings. Do not infer wasm performance from
native TCC measurements.

Proceed to typed bridges only if bridge-free warm execution yields a useful gain
for representative workloads and compile/instantiation cost can be amortized by
the intended reuse pattern. Expensive math calls need a separate decision because
interpreter dispatch may be a smaller fraction of their total cost.

## Validation sequence

1. Host unit tests for source-kind dispatch, eligibility and generated source.
2. Existing full native miniexpr JIT/parity tests to detect host regressions.
3. Emscripten main-module native test with required actual JIT execution.
4. Emscripten side-module test using `me_jit_glue.js`.
5. Exact-i64 boundary and masking differential corpus under Node.
6. Fresh Python-Blosc2 Pyodide wheel and focused public-API tests.
7. Available wasm test suite, then benchmark qualification in at least Node and
   one browser runtime.

Use the `blosc2` conda environment for native Python/build checks in this checkout.
Use the documented Pyodide environment for actual wasm execution; native tests
cannot qualify the wasm ABI.

## Completion criteria for the wasm32 first slice

- At least one float kernel and exact signed/unsigned 64-bit arithmetic kernels
  report and execute an actual wasm JIT kernel through main- and side-module paths.
- No portable wasm source passes through the legacy 64-to-32-bit rewrite.
- Values outside JavaScript's exact integer range round-trip and compute correctly.
- Masks and lazy control flow preserve participation semantics.
- Every unsupported bridge/reduction/ND case demonstrably falls back to the
  interpreter with unchanged results and diagnostics.
- Module imports and the exported entry signature are validated before execution.
- Cache eviction, helper removal and compilation failure release all owned state.
- Existing native host JIT tests and legacy wasm JIT tests remain passing.
- Qualification records distinguish compilation, instantiation, first-call and
  warmed timings and identify tested runtime versions.

## Execution record (2026-10-18)

Implemented in the sibling `miniexpr` checkout:

- Portable Menudet wasm source is explicitly tagged and bypasses the legacy
  64-to-32-bit source rewrite. Its local preamble defines and compile-time checks
  wasm32 fixed-width and pointer widths; portable cache keys include source kind
  and a wasm ABI revision.
- The exported kernel count is `int` (wasm `i32`), with negative counts rejected;
  exact-width input/output values stay in wasm memory. Only zero-bridge,
  non-scalar, non-ND portable elementwise kernels are admitted. Other kernels
  continue through the interpreter.
- Portable modules pass no host bridge lookup and are rejected unless imports are
  their shared `env.memory` or the audited wasm32 `env.memset` primitive. The TCC
  output path is unique per live TCC state and cleaned on success and error paths.
- `tests/test_dsl_jit_portable_wasm32.c` requires actual JIT for signed/unsigned
  64-bit arithmetic and float masking, checks values above `2^53`, verifies an
  unsupported math kernel falls back, and runs a same-workload interpreter/JIT
  timing comparison.

Validation on the local Emscripten SDK with Node.js:

- Environment: Emscripten 5.0.0, Node.js v26.10.0, wasm32 main-module and
  side-module configurations on the local macOS host.

- Main-module and side-module CMake builds and focused CTest both pass. The test
  executes `int64`/`uint64` modular addition, mixed mask lanes, and fallback.
- Native `portable_jit_parity --require-jit --compact` passes all listed cases.
- Initial measurements exposed two generated-code problems: private lane loops
  retained an `int64_t` induction variable despite the wasm32 `i32` entry count,
  and every arithmetic temporary emitted an impossible `pj_status` return branch
  even though first-slice eligibility proves all status-producing bridges absent.
  Using an `int` lane index/count in private wasm helpers and omitting those dead
  wasm-only branches reduced affine JIT time from about 6.2 to 2.8 ms/call.
- After those changes, 1,048,576 float64 lanes of `x * 1.25 + y * 0.75` over 24
  warmed calls took about 3.3 ms/call in the interpreter and 2.6 ms/call under
  TCC JIT (**about 1.3x speedup**) after also removing the same provably-dead
  status checks after local assignment and output store. An eight-term arithmetic
  expression improved substantially but remained slower (about 6.6 ms interpreter,
  7.5 ms JIT).
- Repeating the optimized affine kernel with 10,485,760 lanes and six warmed
  calls gave about 34 ms/call interpreter versus 26 ms/call TCC JIT
  (**about 1.3x speedup**). Cold compile plus module instantiation took about
  2.7–5.0 ms depending on cache/order, so it amortizes in roughly one large call.
- Manually factoring the repeated product in the eight-term case (`p = x*y`, then
  adding `p` eight times) reduced JIT time from about 7.5 to 5.6 ms. This suggests
  a JIT-only common-subexpression pass for pure bridge-free subtrees as the next
  targeted optimization. The corresponding interpreter source uses scalar local
  evaluation, so its 85 ms time is not an apples-to-apples performance baseline;
  compare the factored JIT to the original expression's 6.6 ms interpreter time.
- A separate `-msimd128` Emscripten build confirmed LLVM vectorizes the portable
  batch primitives at 4 float32 or 2 float64 lanes (and 16/8/4/2 integer lanes).
  In Node, SIMD improved the interpreter to about 3.03 ms for affine and 5.28 ms
  for the eight-term expression. The scalar TCC kernels remained about 2.62 and
  7.36 ms respectively, so affine still gained 1.16x, while the unfactored
  arithmetic-heavy case fell to 0.72x. The bounded loop retained about a 2.22x
  JIT gain because it does not use the interpreter's straight-line batch path.
- For a bounded 16-iteration loop over the same 1,048,576 lanes, interpreter was
  about 556–562 ms/call and TCC JIT about 232–243 ms/call (**2.3–2.4x speedup**).
  Compile plus instantiation was about 3.5 ms, and full output buffers matched byte-for-byte.
  A float-comparison branch candidate was not wasm-eligible and correctly used the
  interpreter. These are local Node measurements, not browser or Pyodide results.

Gates at this checkpoint (resolved by the final qualification below):

- The subsequent cross-platform optimization adds conservative structural CSE for
  pure arithmetic subtrees over immutable array inputs/constants, scoped to one
  expression tree, and makes `PJ_STATUS_CHECK()` a no-op when typed lowering has
  registered no status-producing bridge. The latter is target-neutral; wasm's
  32-bit lane counter remains target-specific. On Node, the eight-term repeated
  product improved from 7.53 to 4.70 ms/call (interpreter 6.60 ms, **1.40x JIT**).
- The same benchmark now runs natively on this Apple Silicon host. One isolated
  macOS/arm64 run measured affine at 2.94 ms interpreter / 1.77 ms JIT (**1.66x**),
  while the corresponding wasm32 run measured 3.35 / 2.64 ms (**1.27x**). For the
  eight-term expression, macOS measured 4.01 / 4.72 ms (**0.85x**), while wasm32
  measured 6.60 / 4.70 ms (**1.40x**). The native system interpreter path is
  already very fast; TCC remains slower on that expression even after CSE. These
  are directional local timings, not a controlled cross-machine comparison:
  Node's wasm engine and native TCC have different compile/runtime paths.
- The explicit factored-source control (`p = x*y`) intentionally remains as a
  benchmark: its interpreter path becomes scalar and is not comparable to the
  original vector-batch interpreter. All benchmark cases compare full output
  buffers byte-for-byte before reporting times.
- Acceleration remains workload-dependent: affine and bounded-loop cases benefit
  on both paths; repeated arithmetic now benefits on wasm, but remains slower
  under native TCC. Float-comparison branches remain ineligible on wasm. Keep typed
  bridge families deferred pending broader data-type and browser/Pyodide
  qualification.
- Python-Blosc2/Pyodide integration and a fresh Pyodide wheel were not run at the
  time of the original record; they were completed in the final qualification
  pass described below.
- The original wasm test matrix was a smoke/differential set. It was expanded to
  the boundary, masking, lifecycle, eviction, fallback and browser matrix below.

## Final first-slice qualification (2026-10-18)

`tests/test_dsl_jit_portable_wasm32.c` now runs one differential harness that
prepares every source twice (`jit=False`, `jit=True`) and compares outputs,
including masked lanes. On Emscripten it also asserts the expected route.

Covered in wasm main-module, wasm side-module, and the browser-compatible target:

- signed/unsigned 64-bit `+`, `-`, `*`, unary `-`, shifts, bitwise ops and
  comparisons with `INT64_MIN/MAX`, `INT32_MIN/MAX`, `2^32`, `2^53±1`, `2^63`,
  `UINT64_MAX`, `-1` and `0`;
- float32/float64 `+`, `-`, `*`, `/`, comparisons, signed zero, subnormals,
  infinities and NaNs;
- lazy `where`, nested and deep `where` chains, `if`/`elif`/`else`, bounded
  `for`, `while`, and inline integer floor division;
- mixed/all-active/all-inactive masks, including a zero-divisor lane that must
  not participate;
- zero, one, odd, tile-boundary and 4096-lane lengths;
- process-cache eviction with 70 unique kernels, repeated reuse, forced wasm
  instantiation failure, compile-time width checks, negative/overflow binding
  limits, undersized output/input/mask capacities, overlapping storage, and
  side-module helper unregistration.

A wasm-only defect surfaced and was fixed: `pj_expr` allocated an ~18 KB CSE
table on the wasm stack for each nested lazy expression, so nested `where`
silently exhausted the default stack and fell back to the interpreter. The CSE
table is now heap-allocated and its entry payload reduced; nested/deep `where`
now report an actual JIT kernel. This also lowers stack pressure for all wasm
lowering.

`test_dsl_jit_portable_wasm32_browser` is built with `-sSINGLE_FILE=1` and no
Node raw filesystem. It passes under Node and under headless Chromium for both
main-module and side-module builds, exercising TCC output, module validation,
instantiation and the same differential corpus in a real browser engine.

### Python-Blosc2 / Pyodide

A fresh wheel was built from the local `miniexpr` checkout at
`187707088ae3f9b5e5ff28a03a9fbfc8797b376d` with Pyodide 314.0.7 / Emscripten
5.0.3, installed into a Pyodide venv, and exercised through `@blosc2.menudet`
and `PortableKernel`. `CMakeLists.txt` now pins this exact revision:

- `tests/test_menudet_wasm.py` passes 11 tests covering exact `int64`/`uint64`
  above `2^53`, affine/branch/nested-`where`/bounded-loop kernels, masks via
  `evaluate_block`, tile sizes 1/3/17/1024, repeated reuse, decorator
  `require_jit=True`, and interpreter fallback for math/predicate/cast/
  mixed-comparison/reduction kernels.
- `blosc2._WASM_MINIEXPR_ENABLED` is true and `PortableKernel.has_jit` reports
  actual kernels, not merely interpreter parity.

Warm Pyodide timings for 1,048,576 float64 lanes (10 calls):

| Kernel | Interpreter | TCC JIT | Speedup | Cold JIT setup |
|---|---:|---:|---:|---:|
| Affine | 3.34 ms | 1.73 ms | 1.93x | 6.4 ms |
| Eight-term product | 4.99 ms | 2.09 ms | 2.39x | 1.7 ms |
| 16-iteration loop | 47.54 ms | 2.31 ms | 20.59x | 1.7 ms |
| Branch | 62.45 ms | 1.87 ms | 33.47x | 1.8 ms |

### Differences from the original plan

- The zero-copy contiguous `me_artifact_eval_ex` ABI requires naturally aligned
  input/output storage and rejects offset pointers as binding errors. This is not
  a user-facing Python limitation: `PortableKernel` copies/normalizes inputs with
  C-contiguous aligned buffers, and the array-view adapter supports offset,
  stepped, transposed and reversed views with bounded tile gathering. An
  unaligned NumPy `float64` view (address `% 8 == 1`) was verified through both
  interpreter and JIT Python paths. The wasm test's negative case covers only the
  contiguous fast ABI; unaligned pointers are never passed to generated typed
  loads.
- The side-module helper ABI does not carry the test-only
  `ME_WASM_JIT_TEST_FORCE_INVALID` flag. Main-module builds corrupt the module to
  exercise validation failure; side-module builds register a deliberately
  rejecting instantiate helper and verify cleanup/interpreter fallback, then
  restore the real helper. Helper unregistration is tested separately.
- Concurrent compilation is not applicable to this single-threaded build: the
  wasm output does not enable pthreads or shared memory. The process cache is
  still exercised through eviction and repeated reuse.
- Several operations originally expected to need bridges are lowered inline on
  wasm: ordinary float comparisons, literal shifts, integer floor division and
  nested lazy `where`. They now run through the bridge-free JIT. Math functions,
  predicates, checked float-to-integer casts, mixed signed/unsigned comparisons
  and reductions remain interpreter fallbacks.
- Pyodide tests run in the Pyodide venv under Node. Browser qualification uses
  the Emscripten wasm test directly; Pyodide hosted in a browser was not run.
- Floating-point status flags are not observable on wasm (`supported=0`), so
  qualification compares values and bit patterns (NaN category where payloads
  may differ) rather than IEEE flag parity.

Remaining work is the deferred Task G typed-bridge ABI, plus optional
browser-hosted Pyodide and wider workload/dtype/performance sweeps. No
first-slice correctness or lifecycle blocker remains.

## Non-goals

- Enabling SLEEF, Accelerate, relaxed floating point, SIMD, threads, or every math
  operation on wasm32.
- Replacing the Menudet interpreter or removing best-effort fallback.
- Passing exact integers through JavaScript `Number`.
- Treating native TCC speed as evidence of wasm32 speed.
- Broadening portable semantics, changing NumPy compatibility, or changing the
  public meaning of `jit=True`.
