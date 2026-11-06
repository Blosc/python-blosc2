# Flexible math libraries and floating-point modes for Menudet

Status: implemented 2026-10-18 (flexible source/env math selection, artifact
policy, per-kernel dispatch, interpreter/vector and CC/TCC integration).
Numerical qualification probes and reports predate this integration; see
[integrated performance results](mathlib-performance.md) for a measurement that
does not imply numerical certification.

## Objective and scope

Allow Menudet kernels to select system math, SLEEF or Apple Accelerate explicitly,
independently of compiler floating-point optimizations. Preserve the current
system-math behavior by default. Give the legacy SIMD path the same environment
selection policy while it remains supported, rather than maintaining conflicting
meanings ahead of its eventual removal.

Implement the policy in native miniexpr, then integrate its artifacts and Python
bindings in Python-Blosc2. Native numerical semantics and backend matrices belong
in miniexpr tests. Python tests should cover authoring, serialization, errors,
dispatch, scheduling and representative end-to-end execution.

Out of scope: removing legacy SIMD now; adding new mathematical operations;
general numerical repair wrappers around third-party libraries; wasm32 TCC JIT;
new dependencies beyond existing optional SLEEF/Accelerate support. The wasm32
JIT investigation is tracked separately in
[todo/enable-tcc-menudet-wams32.md](../todo/enable-tcc-menudet-wams32.md).

## Agreed user-visible behavior

### Math selection

| Effective selection | Behavior |
| --- | --- |
| `auto` | System math with Menudet's existing operation definitions. No automatic SLEEF/Accelerate preference. |
| `system` | Explicit system-math policy. This does not disable JIT or safe execution optimizations. |
| `scalar` | Retain the existing environment spelling for forced system scalar math. |
| `sleef` | Explicit SLEEF selection; default accuracy u35, optionally u10. |
| `accelerate` | Explicit Accelerate selection on supported macOS builds. |

`ME_SIMD_MATH_BACKEND` remains the environment selector. Add `system` as an
accepted value and change legacy `auto` to the same system policy as Menudet.
Keep `scalar` as its existing explicit scalar option; the source pragma need
not add a duplicate spelling. Do not rename the environment variable in this
effort, despite its historical SIMD-specific name.

An available accelerated backend may use documented system fallbacks for
operations it does not implement. Backend availability and operation coverage
are different checks. Do not silently replace an explicitly requested library
with another library when that backend itself is unavailable.

### Source pragmas

Follow the existing header-pragmas convention:

```python
# me:compiler=cc
# me:fp=fast
# me:math=sleef
# me:ulp=u35
def kernel(x):
    return sin(x) + cos(x)
```

- `me:math=auto|system|sleef|accelerate`.
- `me:ulp=u10|u35`, optional; SLEEF defaults to u35.
- Existing `me:compiler=tcc|cc` remains independent.
- Enable `me:fp=strict|contract|fast` for Menudet as described below.
- Reject unknown values, duplicate keys and malformed directives consistently
  with existing pragma parsing. Preserve source locations in errors.
- Validate `me:ulp` against the resolved math backend. It is meaningful only
  for SLEEF; reject it when the effective backend is not SLEEF instead of
  silently ignoring it. With no math tag, an environment-selected SLEEF backend
  may supply that effective selection.

### Precedence and resolution lifetime

For each setting: explicit source pragma, then applicable environment setting,
then built-in default. Preserve an explicit-presence bit rather than inferring
presence from a parsed enum's default value.

- No math tag: inherit `ME_SIMD_MATH_BACKEND`, or `auto` if unset.
- Explicit `me:math=auto`: select the defined system policy; do not consult the
  environment for math selection.
- Explicit `me:math=system` overrides an environment request for Accelerate,
  including on a platform where Accelerate is unavailable.
- Explicit `me:fp` overrides `ME_DSL_FP_MODE`; absent either, use strict.
- Preserve the existing compiler source-over-environment precedence.
- Do not introduce a new environment variable solely for SLEEF accuracy in
  this work. Reuse existing native per-call accuracy controls where applicable;
  source accuracy must be respected consistently rather than bypassed by them.

Resolve defaults once per prepared kernel, before first execution, and retain
an immutable effective policy for its lifetime. Reading process environment
variables inside worker loops is neither necessary nor safe. Different prepared
kernels must coexist with different explicit policies in one process.

### Floating-point modes

| Mode | CC flags / permission | Compatibility |
| --- | --- | --- |
| `strict` | `-fno-fast-math -ffp-contract=off` | Default compiler behavior, current operation semantics. |
| `contract` | `-fno-fast-math -ffp-contract=fast` | Permit FMA contraction and resulting rounding/status differences. |
| `fast` | `-ffast-math` | Permit broader transformations; strict NaN/infinity/zero/rounding/status compatibility is waived. |

Math-library choice and FP mode are independent. In particular, strict plus
SLEEF/Accelerate is permitted: strict constrains generated arithmetic, but does
not upgrade the selected library's documented values, domains or exceptions.
Fast plus system math and fast plus SLEEF/Accelerate are also valid.

Native TCC currently supports strict only. Explicit TCC plus contract/fast must
give a clear unsupported-combination error, not silently execute as strict.
Optional interpreter execution may retain ordinary arithmetic under relaxed FP
policy: relaxation grants optimization permission rather than requiring a
particular changed result. Required-JIT and explicit unsupported-compiler
requests must retain clear failure behavior; do not hide them with fallback.

### Numerical contract

Preserve operation identity across backend selection: floor-mod remainder,
ties-even rounding and other defined Menudet operations must not accidentally
become a same-named C function with a different definition. Review pi functions
and min/max separately, using existing semantic helpers where appropriate.
Under fast FP, the user separately accepts compiler-level numerical relaxation;
do not claim strict signed-zero/NaN behavior for optimized generated arithmetic.

For explicitly selected SLEEF/Accelerate functions, accept their documented
accuracy, domain restrictions and exception behavior. Do not construct a broad
adapter to repair every backend result, signed zero or floating exception flag.
Changing u35 to u10 improves selected approximation bounds, not these contracts.

Document exception reporting honestly: caller environment isolation/preservation
remains required, but NumPy-equivalent status from relaxed math libraries or
fast arithmetic is not guaranteed. Keep integer semantics, types, bounds,
mask participation and memory safety independent of numerical relaxation.

### Platform and failure policy

- `auto`/`system`: available on Linux, macOS, Windows and wasm32; wasm32 uses
  the runtime/toolchain math implementation.
- `sleef`: require both a supported compiled implementation and target CPU
  capability. Do not assume availability merely from an OS name. Current native
  dispatch includes AVX2 and ARM Advanced SIMD; wasm32 support is not established.
- `accelerate`: require macOS and an Accelerate-enabled build.
- Unavailable effective explicit backend requests from source or environment
  fail during preparation, before evaluating any data. Report backend, platform
  and supported alternatives. Validate malformed effective environment settings
  rather than silently treating misspellings as auto.
- Individual documented system fallbacks within an available backend remain
  valid. They must be visible in test/diagnostic information.

## Current implementation and gaps

Relevant native code in the sibling miniexpr checkout:

- `src/dsl_parser.c`: currently understands only `me:fp` and `me:compiler`.
- `src/dsl_compile.c`: typed portable profiles explicitly reject non-strict FP.
- `src/dsl_portable_expr.c`: system callbacks and Menudet semantic operations.
- `src/dsl_portable_batch.c`: bounded vector interpretation, currently strict.
- `src/dsl_portable_jit.c`: typed generated kernels and strict host bridges;
  wasm32 source generation is explicitly declined for ABI reasons.
- `src/functions-simd.c`: existing SLEEF/Accelerate wrappers, process-wide backend
  initialization/function pointers, and thread-local scalar/accuracy overrides.
- `src/dsl_config.h` and `src/dsl_jit_backend_*.c`: existing FP flags, environment
  defaults and strict-only native TCC behavior.
- `src/dsl_artifact.c`, artifact headers and JIT cache code: validated artifact
  representation, capability requirements, loading, preparation and cache identity.

Python integration:

- `src/blosc2/dsl_kernel.py`: source retention and `DSLKernel.export()`.
- `src/blosc2/portable_kernel.py`: normalized source and artifact construction;
  currently emits `semantics.fp = strict` unconditionally.
- Native/Cython bindings and build pins must expose the matching native feature
  and error contracts. Locate all artifact preparation/evaluation callers before
  changing any native ABI or lifecycle.

Build configuration already enables Accelerate on macOS and dynamically links
the system framework. No Apple implementation is bundled; enabling Menudet
dispatch does not require shipping a large additional library.

## Implementation sequence

### 1. Define and test policy resolution in miniexpr

Add the minimal native policy representation: requested math backend, requested
SLEEF accuracy, FP mode, explicit-presence information, and effective settings.
Reuse existing enums where their meanings fit. Separate backend availability
from operation coverage. Make one resolver authoritative for typed preparation
and legacy selection rather than duplicating precedence in Python.

Add parser support and native unit tests for all values, duplicate/unknown
directives, omitted versus explicit auto, environment precedence and invalid
combinations. Test environment cases in isolated subprocesses where necessary.
Specify precedence of existing per-call overrides explicitly before wiring them:
they must not silently weaken an explicit kernel numerical policy.

### 2. Persist executable policy in artifacts

Use validated semantic/execution fields, not informational `metadata`.
Preserve explicit source choices through export, JSON round trips and native
loading; retain omitted settings as inheritance requests resolved at preparation.
Do not bake one machine's environment-derived choice into exported source.

The current artifact schema rejects unknown fields and requires strict FP.
Extend its version/capability contract deliberately: introduce a recognized
flexible-math requirement for artifacts depending on the new policy, update
the schema specification, and require old readers to fail closed. Settle exact
wire-field names and whether a draft version increment is necessary before
coding the exporter. Do not reinterpret an existing strict claim as fast.

Validate source directives against declared artifact policy and reject conflicts.
Keep old artifacts without new choices on their documented interpretation;
explicit legacy strict FP declarations remain strict. Document whether missing
math policy inherits the new environment default, consistent with omission rules.

### 3. Make backend dispatch kernel-local

Refactor only as much of `functions-simd.c` as needed to reuse its wrappers via
an immutable per-prepared-kernel table/descriptor. Do not mutate process-global
function pointers when alternating kernels. Resolve CPU/build availability once
and pass the selected callbacks through existing typed runtime bridges.

Give legacy `auto` the new system meaning, add the system spelling, and enforce
explicit-unavailable errors. Retain the scalar setting and existing per-call
SIMD-disable/accuracy behavior where compatible with explicit kernel policy.
Keep user-visible function definitions separate from raw library entry points.

### 4. Integrate Menudet interpretation and bounded vector execution

Use the same resolved math selection in scalar interpretation and eligible
bounded vector interpretation. Audit scalar tails and partial masks: a SLEEF
kernel must not inadvertently change library merely because its array length
falls below SIMD width. If wrappers lack scalar SLEEF entry points, use bounded
lane packing or another minimal safe bridge; document and test any intentional
fallback instead of inheriting legacy tails unnoticed.

Preserve typed float32 calls and signaling-NaN transport where the contract
requires it. Maintain operation semantic helpers, inactive-lane isolation,
caller fenv preservation and bounded scratch. Do not aggregate exceptions from
speculative inactive lanes as if the source evaluated them.

### 5. Integrate TCC/CC generated kernels and relaxed FP

Route generated calls through the same prepared callback table/bridge. Preserve
bounded tiling rather than adding full-array temporaries. Keep backend policy
valid for JIT-disabled and optional-fallback execution as well as warmed JIT.

Remove the blanket portable non-strict rejection only after both interpreter
policy and generated-code contracts are implemented. Apply mode flags to the
generated arithmetic, not to the entire library or safety helpers. Review
generated NaN/finite checks, floating-to-integer conversions, index/bounds checks
and optimizer assumptions; move safety-critical checks behind strict helpers
where necessary. Existing strict-only generated-source guards must become
mode-aware rather than simply being deleted.

Native TCC: implement strict math-backend bridges and reject explicit relaxed
FP requests. CC: support strict, contract and fast. Keep wasm32 on its current
interpreter path; do not imply a new wasm32 JIT capability.

### 6. Cache and concurrency correctness

Include effective math backend, SLEEF accuracy, FP mode, bridge ABI and relevant
implementation/version identity in JIT eligibility/cache keys and validation.
Keep system/auto normalization consistent; never reuse incompatible kernels.
Do not serialize function addresses. Re-resolve available runtime callbacks on
load. Platform-dependent artifacts fail clearly when explicit requirements
cannot be met on the destination machine.

Exercise concurrent and interleaved kernels with different backends/modes,
thread-pool execution, cache hits and cache invalidation. Preserve the existing
serialization needed by libtcc compilation; runtime callbacks must remain
kernel-local and safe after compiler state cleanup.

### 7. Python integration, documentation and migration

Update exporter/bindings to preserve validated pragmas and artifact policy, and
translate native preparation errors into the existing structured error types.
Keep policy ownership native; avoid a second Python backend-selection algorithm.
Cover `DSLKernel.from_source`, decorated-source extraction, artifact export/import,
`PortableKernel`, lazy execution and evaluate-generated artifacts as applicable.

Update native DSL syntax/usage, artifact specification and Python docs. Provide
strict-system, strict-SLEEF-u35, CC-fast-system and CC-fast-SLEEF examples. Explain
precedence, unavailable libraries, system fallback coverage and platform limits.

Call out two legacy behavior changes: auto no longer prefers SLEEF, and an
unavailable explicit backend now errors. Existing users wanting the old SLEEF
selection can request it explicitly. Do not remove legacy evaluation in this
change. Update the native dependency pin and Python build configuration only
after the native implementation is independently qualified.

## Validation and acceptance

1. Native parser/resolver/artifact tests establish source-over-environment
   precedence, effective choices, schema round trips and fail-closed imports.
2. Native numerical coverage distinguishes operation definitions from backend
   approximation/domain/exception behavior. Reuse the existing qualification
   corpus and reports; do not duplicate exhaustive matrices in Python tests.
3. Strict-system interpreter/TCC/CC preserves current supported semantics.
   Explicit SLEEF/Accelerate calls demonstrably reach the chosen implementation
   on full vectors, short arrays and tails. System fallback cases are identified.
4. Contract tests demonstrate a suitable FMA-sensitive example without assuming
   contraction is mandatory. Fast tests verify flags/mode application and safety
   invariants, rather than demanding one compiler-specific approximate result.
5. Bounds, integer arithmetic, mixed dtypes, masks, exceptions, reductions, empty
   inputs and concurrency retain their non-relaxed guarantees. Explicit invalid
   combinations and unavailable backends fail before execution.
6. Run the common 44-function/input audit against actual integrated Menudet paths,
   covering u10/u35/Accelerate and separate FP modes. Preserve reproducible input
   hashes and independent references; raw ULP-versus-system differences are not
   automatically backend defects. Broaden ldexp/fma separately as appropriate.
7. Benchmark cold preparation/JIT and warm execution against current strict-system
   behavior and NumExpr: arithmetic, transcendental-heavy and mixed kernels,
   float32/64, small and large arrays, single/multiple threads, TCC/CC. Do not
   infer performance from the numerical audit. Check scratch and binary size.
8. Qualify Linux and macOS first, Windows on its supported build/CPU configuration,
   and wasm32 interpreter behavior. Unsupported SLEEF/Accelerate requests must
   error on the applicable targets. A missing runner is an explicit validation
   gap, not a passed platform.

Use the `blosc2` conda environment for Python, tests and builds. Reuse incremental
native builds with the same miniexpr source override, verify loaded source and
extension paths, and use isolated packages where needed rather than replacing
another checkout's editable extension. Finish with relevant native checks,
Python binding tests and an independent wheel check. No git state-changing
operations are part of this plan without explicit user instructions.

## Implementation outcome (2026-10-18)

Implemented in native miniexpr and Python-Blosc2:

- `me:math`/`me:ulp` parsing, presence bits and one resolver with source >
  artifact > environment > default precedence; invalid env values fail.
- Artifact `semantics.math`/`semantics.ulp`, optional `semantics.fp`, and the
  `flexible-math` capability with fail-closed old-reader behavior.
- Kernel-local immutable callback tables; legacy `auto` now means system math.
- Bounded vector interpretation uses accelerated tables over tiles; scalar
  interpretation uses documented lane packing.
- `contract`/`fast` reach CC-generated kernels; TCC rejects them explicitly.
  Portable artifacts with accelerated math deliberately use the bounded vector
  interpreter instead of the fused per-lane JIT, whose callback bridge would pay
  one lane-packing dispatch per element.
- JIT cache identity includes math backend and SLEEF accuracy.
- Python export/info/tests and documentation updated.

Deliberate differences from this plan: accelerated math does not use generated
JIT kernels; it uses the vector interpreter fallback described above. SLEEF
accuracy is not exposed through a new environment variable (per plan). The legacy
SIMD path still uses its process-global dispatch rather than the new per-kernel
table, but its `auto`/`system` meaning and preparation-time availability errors
now agree with Menudet.

## Existing evidence and remaining risks

- [Shared backend audit](math-backend-shared-qualification.md) and
  [raw results](math-backend-shared-qualification.json): same 44 functions,
  float32/64 and input hashes across u10/u35/Accelerate configurations.
- [Broader SLEEF audit](sleef-menudet-qualification.md): documented range limits,
  operation differences, flags and independent-reference checks.
- [Accelerate audit](accelerate-menudet-qualification.md): value/status behavior
  and Apple's explicit exception/denormal caveats.
- [Performance exploration](math-backend-performance.md): prototype measurements,
  not a qualification of integrated generated Menudet execution.

Highest risks: global backend state leaking across kernels; scalar tails using
different semantics; old artifacts silently gaining relaxed behavior; fast-math
weakening safety checks; inappropriate exception guarantees; and unsupported
platform assumptions. The wire-format extension and compatibility tests must
be settled before enabling any new execution mode by default.
