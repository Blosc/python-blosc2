# Menudet: strong NumPy compatibility implementation plan

Date: 2026-10-09. Branches: `numpy-compat` in Python-Blosc2 and miniexpr.
Status: **M1–M6 declared-subset implementation substantially complete; release
acceptance incomplete.** Cross-platform CI qualification is in progress and M7 scope
review has started; no semantic freeze, release/default-backend change or expanded
packaging promise has been approved.

### Current milestone position (2026-10-09)

| Milestone | Position | Remaining acceptance work |
| --- | --- | --- |
| M1–M2: specification / conformance | Implemented checkpoints and authoritative shared corpora. | Consolidate historical inventory into a current release matrix/register. |
| M3: arithmetic / promotion / casts | Declared eleven-dtype matrix implemented in opt-in 1.1. | Exact-revision cross-platform/build qualification; retain explicit divergences. |
| M4: functions / floating policy | Enumerated real signatures and native per-call status implemented. | Platform/libm qualification; no full `np.seterr` promise. |
| M5: logical arrays | Native descriptors, broadcasting, layouts, shape operations and six reductions implemented. | Platform qualification; comprehensive/end-to-end compressed memory claims remain unproven. |
| M6: graph integration / performance | Native-required subset, plan reuse, persistence and opt-in host JIT implemented. | Integrated performance budgets, clean-install packaging audit or approved deferral, distribution qualification. |
| M7: freeze / release | Scope review started; proposed contract and open decisions recorded. | Approve scope, close CI/build/docs gates, freeze and publish. |

Current review: `plans/menudet-m7-scope-review.md`. Implementation evidence is in
the M1/M2, M3, M4, M5/M6 and expanded-JIT signoff reports; their dates/revisions
matter. Python `6c863eba` / native `2567eb6` are the expanded-lowering qualification
baseline. Native HEAD at this review is `25a4aa4`, with additional lowering work;
baseline results do not automatically qualify those later commits. CI findings
are recorded in `plans/menudet-ci-qualification.md`: Linux linkage, Windows builtin
identity/remainder and WASM fixture fixes are implemented locally. Patched native
`25a4aa4` with Python `6c863eba` passed 430 native, 63 WASM and 483 explicit-pair
Python tests (zero Python skips). These working-tree results are not clean-SHA or
remote Linux/Windows qualification; the fixes remain unpublished and the current
pair's remote qualification remains open.

The sections below retain the original acceptance criteria. Proposed M7 deferrals
(including NumExpr optional packaging) do **not** silently mark those criteria
complete. Artifact language/schema 1.1 is distinct from product release numbering;
checked 1.0 artifacts and default routes remain unchanged.

## 1. Objective and release strategy

Make miniexpr/Menudet capable of reproducing a useful and progressively broader
subset of NumPy numerical expressions from other programming languages, without
Python in the execution loop. Preserve Menudet's additional programming facilities:
local variables, branches, bounded loops, typed captures and logical coordinates.

NumPy is the semantic reference. NumExpr is an existing execution backend and a
useful performance comparator, not the authority for promotion or numerical results.
NumExpr itself demonstrates that a useful engine need not reproduce all of NumPy.

Develop compatibility incrementally. Stop extending the 1.0 scope when an extension's
implementation, performance or maintenance cost outweighs its practical value.
Record the remaining divergences and unsupported operations, freeze the resulting
contract, and release 1.0 with a precise compatibility statement. Full NumPy
compliance is neither a prerequisite nor an implied promise.

Menudet 1.0 is unreleased. Its existing checked-arithmetic and promotion decisions
may be revised deliberately toward NumPy behavior. We do not need two permanent
arithmetic personalities merely to preserve a draft decision. Separate profiles or
explicit checked operations should be introduced only for a demonstrated use case.

### Relationship to the safer-LazyExpr experiment

The safer-LazyExpr experiment is successful in its architectural objective:
validated graphs and explicit dispatch provide an eval-free path for the supported,
tested operations while preserving the existing lazy execution machinery. Its
platform verification is green at `063fd8b9`; see
`plans/safer-lazyexpr-results.md` for the exact evidence and boundaries.

That is our foundation, not a claim of universal safety, default-switch readiness,
complete native allocation bounds or same-storage read/write synchronization.
The NumPy-compatibility work is a new numerical-semantics project. Earlier plans
that deliberately excluded changes to Menudet arithmetic describe the scope of the
previous experiment; they do not prohibit deliberate draft revisions here.

Reuse the graph, metadata, persistence, differential-testing and benchmarking
machinery. Preserve ordinary LazyExpr behavior while the new backend is opt-in;
NumPy compatibility of a new mode is not permission to silently change the results
of every existing NumExpr/Blosc2 expression. Default adoption is a separate decision.

## 2. Compatibility claims and initial scope

Track compatibility separately for:

1. **Values:** integer/Boolean exactness and floating numerical agreement.
2. **Types:** input acceptance, promotion, intermediate types and output dtype.
3. **Shapes:** broadcasting, output shape, axes and reduction cardinality.
4. **Exceptional behavior:** NaNs, infinities, signed zero, overflow and errors.
5. **Diagnostics:** native error/status categories and floating exception reporting.
6. **Storage behavior:** layout, copying, views, aliasing and output buffers.
7. **Execution quality:** fusion, memory use, parallel scheduling and performance.

An operation may match the first five without reproducing NumPy views or allocation
patterns. Do not describe a dtype mismatch as full compatibility merely because
small sample values compare equal. Likewise, a native diagnostic category need not
have NumPy's Python warning class or exact message to be useful in a C host.

### Reference policy

- Start with one pinned NumPy 2.x release. Proposed baseline: **2.5.3**, the native
  version used in the completed experiment; confirm this in milestone 1.
- Record NumPy version, platform integer width, endianness, Python scalar category
  where applicable, floating-error policy and reference-generator revision.
- NumPy 1.26 remains a Python-Blosc2 host compatibility target, not a second
  simultaneous definition of Menudet arithmetic. A pinned corpus must not change
  its meaning according to whichever NumPy version happens to be installed.
- Run secondary NumPy versions to identify reference drift; do not silently
  regenerate golden results or change the declared baseline.
- Specify the interpretation of `int`/`intp` and omitted reduction dtypes on 32-bit
  versus 64-bit hosts. Prefer explicit widths in portable artifacts. Decide how
  source-platform defaults are resolved rather than inheriting the C compiler's
  `long` width or accidentally changing persisted results on WASM.

### Initial priorities

Prioritize Boolean, standard signed/unsigned integers, float32 and float64; typed
elementwise operations; promotion/casts; common real functions; broadcasting; and
sum/product/min/max/Boolean reductions. Existing string and control-flow capabilities
remain supported but do not imply NumPy string/object compatibility.

Initially defer complex arithmetic, object arrays, arbitrary Python callbacks,
arbitrary memory access, structured/object semantics, datetime/timedelta calendars,
generalized ufuncs, linear algebra, advanced indexing and full view/aliasing parity.
Assess float16 and extended-precision floats explicitly as dtype extensions rather
than treating their current absence as accidental test omissions.

## 3. Cross-repository architecture and ownership

### miniexpr owns native semantics

- Normative type, operator, cast, function and exceptional-value rules.
- Validation, type inference, interpreter execution and native diagnostics.
- Artifact representation and semantic capability/version identification.
- JIT/SIMD eligibility and agreement with the interpreter.
- Native conformance runners and tests independent of Python at execution time.

Initial implementation inventory includes `src/dsl_portable_types.c`,
`src/dsl_portable_expr.c`, `src/dsl_portable.c`, their headers,
`src/dsl_portable_fp.h`, and the full DSL compilation path. Existing native tests
include `tests/test_dsl_portable_types.c`, `test_dsl_portable_validation.c`,
`test_dsl_portable_interp.c` and `test_dsl_portable_checked.c`. Confirm actual
dispatch routes before editing; these are starting points, not an exhaustive map.

### Python-Blosc2 owns authoring and integration

- Preserve scalar/type intent when lowering Python authoring or expression graphs.
- Bind arrays, storage and logical partitions without redefining native arithmetic.
- Expose native capabilities and diagnostics through the Python extension.
- Integrate lazy scheduling, reductions, persistence and backend eligibility.
- Generate NumPy reference cases and run differential/integration/performance tests.

Principal integration points: `expression_graph.py`, `lazyexpr.py`, `dsl_kernel.py`,
`portable_kernel.py`, `portable_lazy.py`, `blosc2_ext.pyx`, `b2objects.py` and CTable
kernel registration. Reuse the existing math-certification scripts where their
reference and comparison policy apply; NumPy parity and mathematical accuracy are
separate measurements.

### Native array orchestration is a distinct layer

NumPy broadcasting and axis reductions cannot be provided to a C caller merely by
implementing them in Python's `PortableLazyArray`. Milestone 5 must place the
necessary in-memory traversal/grouping machinery behind a native API, with a clear
boundary between miniexpr and C-Blosc2 where compressed storage is involved.
Kernel semantics must not depend on Python-Blosc2 being installed.

### Paired revision discipline

- Record starting SHAs for both branches and the exact native dependency revision
  used by every integrated result. Branch names alone are not reproducible pins.
- Land native support first, then the matching bindings/lowering/integration.
- Pin native dependencies through the established build configuration; do not
  silently search a sibling checkout or rely on an old locally compiled extension.
- Keep the native corpus authoritative and share/version its schema with Python;
  avoid maintaining two independently edited semantic rule sets or golden corpora.
- Changes to ordinary full miniexpr execution, portable Menudet execution and the
  Python API must be identified separately, even when they share implementation.

## 4. Milestone 1 — specification, inventory and baseline

### Implementation work

1. Confirm the NumPy reference release and supported target platforms.
2. Trace existing promotion and execution in full miniexpr, portable interpreter,
   JIT/SIMD paths and Python bindings. Identify output-dtype-driven computation,
   implicit scalar coercion and host-width-dependent defaults.
3. Create a machine-readable capability matrix plus a human-readable summary.
4. Establish baseline examples for arithmetic, casts, scalar/array promotion,
   exceptional values, functions, broadcasting and reductions.
5. Create the divergence register described below; seed it with checked overflow,
   promotion differences, missing complex/half types, function gaps, block-local
   versus axis reductions, and adapter copying/binding restrictions.
6. Record performance baselines before semantics or lowering changes.

Each matrix row needs an operation/signature, input categories/dtypes, reference
configuration, expected output type/shape, participating backend, test IDs and one
of `matching`, `divergent`, `unsupported` or `unverified`.

**Acceptance:** current behavior is reproducibly classified; initial 1.0 priorities,
reference version and comparison policies are explicit. No inventory item becomes
supported simply by appearing in a proposed milestone.

## 5. Milestone 2 — language-independent conformance machinery

### Portable reference format

Define a versioned vector format with stable case IDs and:

- Operation graph or normalized kernel, semantic revision and required capabilities.
- Input names, explicit dtype descriptors, shapes, scalar strength/type and layout.
- Integer values represented losslessly; IEEE float bits encoded losslessly;
  signed zeros and exceptional classifications retained. Avoid JSON-number loss.
- Expected output dtype, shape and values, or expected native diagnostic category.
- Exact/bitwise/tolerance comparison policy, with explicit absolute/relative/ULP
  thresholds and separate NaN/signed-zero treatment.
- Reference provenance, generation seeds and deliberate divergence annotations.

Represent scalar-versus-0-D-array distinctions explicitly. Define byte order and
layout recipes so C and Python readers reconstruct identical inputs. NaN payload
preservation should not be inferred from an ordinary equal-NaN comparison.

### Runners and generation

1. A pinned Python/NumPy generator emits reviewed vectors; it is development tooling.
2. A native runner executes checked-in vectors without Python/NumPy installed.
   Reuse existing native JSON/test infrastructure where practical.
3. A Python runner compares NumPy, native interpreter, eligible JIT paths and
   Python-Blosc2 integration on the same case IDs.
4. Add seeded property tests and failure minimization; promote important minimized
   failures into stable vectors rather than relying only on random coverage.
5. Record skipped capabilities and actual backend selection. A requested JIT that
   falls back successfully is interpreter evidence, not JIT conformance evidence.
6. Verify diagnostic cleanup and repeated execution after failures.

Keep finite mathematical-reference certification alongside NumPy comparisons.
Agreement with one NumPy/libm build is not proof of mathematical correctness over
all inputs. Do not inflate accuracy tolerances merely to turn a platform green.

**Acceptance:** the same corpus runs in a standalone native host and Python; reports
identify every mismatch and skip. Include at least one small C example consuming
an artifact and explicit typed buffers with no Python runtime dependency.

## 6. Milestone 3 — arithmetic, promotion and casts

Implement in vertical slices: specification, reference vectors, native interpreter,
validation/inference, eligible acceleration, Python lowering and persistence tests.

### 3A. Fixed-width arithmetic

- Boolean, signed and unsigned arithmetic and comparisons for supported widths.
- Integer wraparound where NumPy defines fixed-width array arithmetic that way.
- Unary negation and absolute-value boundary cases, including signed minima.
- True division, floor division and remainder, especially negative operands.
- Bitwise operators and shifts, with explicit invalid/large shift-count treatment.
- Integer zero divisors, minimum divided by minus one and other exceptional cases.
- Floating arithmetic with specified intermediate precision and contraction rules.

Implement signed wraparound through well-defined bit/unsigned operations and
explicit conversion rules, never C signed-overflow undefined behavior. Audit
vectorized/JIT code separately. Numerical wrapping does not remove array bounds,
artifact-validation or resource checks.

### 3B. Promotion and scalar categories

- An enumerated mixed-dtype promotion matrix, including signed/unsigned mixtures.
- NumPy 2.x weak scalar behavior versus explicitly typed scalar operands.
- Literals and captures whose values do not fit the selected operand dtype.
- Python scalar, typed scalar and zero-dimensional array differences, represented
  as language-independent categories rather than references to Python objects.
- Comparisons and predicates whose promotion differs from arithmetic operations.
- Intermediate types computed from operations and operands, not forced by the
  requested output buffer or final result conversion.

Decide syntax/API and artifact encoding for scalar categories before extending
authoring. Preserve type intent through export/import and across non-Python hosts.

### 3C. Explicit conversion

- Widening/narrowing integers, float precision changes and Boolean conversions.
- Float-to-integer truncation, range boundaries, NaNs and infinities.
- Supported cast policies corresponding to NumPy's safe/same-kind/unsafe concepts.
- Out-of-range construction versus array arithmetic versus explicit conversion:
  these are different NumPy operations and need separate rules/tests.

Where NumPy results depend on hardware/compiler details or are not a stable
documented contract, choose a defined behavior or reject and record the divergence.
Do not codify one machine's incidental result as a universal standard.

**Acceptance:** the declared dtype/operator/cast matrix passes on interpreter and
eligible JIT paths across native platforms and WASM. Promotion agrees with actual
execution and remains metadata-only. Unsupported combinations reject predictably.

## 7. Milestone 4 — functions and exceptional-value semantics

Prioritize common arithmetic helpers and real math, then add functions by measured
usefulness. Audit aliases and signatures as well as function availability.

Test `minimum`/`maximum` separately from `fmin`/`fmax`; rounding variants, sign,
absolute value, `where`, power, logarithms, exponentials, trig/hyperbolic operations,
classification and existing nextafter/copysign-style primitives.

For each function cover:

- Accepted input types, promotion and result dtype.
- Finite values, domain boundaries, cancellation and large/small magnitudes.
- NaNs, infinities, positive/negative zero and subnormals.
- Domain, overflow, underflow and divide-by-zero reporting.
- Evaluation of unselected `where` branches and observable diagnostics.
- Float32 evaluation versus computing in float64 and narrowing afterward.
- Interpreter, SIMD and JIT agreement with the declared accuracy policy.

Design a native floating-status/error policy usable from C and other languages.
Define clearing/aggregation across blocks, masking participation, threading and
restoration of the caller's floating environment. Initially document differences
from `np.seterr` rather than promising its complete Python callback/warning API.

**Acceptance:** every advertised function has signature and exceptional-value
coverage plus explicit accuracy criteria. Any unsupported NumPy spelling or
different numerical rule is visible in the matrix and divergence register.

## 8. Milestone 5 — array semantics and native scheduling

### 5A. Elementwise traversal

Define native descriptors for typed buffers, shape, strides, byte order and output
ownership, with overflow-checked extent/offset calculations. Add broadcasting for
scalars, 0-D arrays, singleton dimensions and empty dimensions. Cover C/F layouts,
noncontiguous and negative-stride views, or reject/document unsupported layouts.

Separate a functional copying adapter from an optimized zero-copy iterator. Report
normalization allocations so layout compatibility does not conceal full-array copies.
Define allowed output-buffer overlap before accepting in-place/aliased execution.

### 5B. Reductions

Start with sum/prod/min/max/any/all; assess mean, variance/std, arg reductions and
cumulative operations as subsequent slices, not implicit commitments for 1.0.

Specify axis normalization, selected-axis combinations, `keepdims`, accumulator
dtype, empty identities/errors, initial values and participating masks. Preserve
the distinction between a logical array reduction and Menudet's explicit block
reduction. Record grouping where it is semantically observable.

Design native traversal/combination for logical reduction groups independent of
compressed storage chunks. Floating reduction order may differ from NumPy's
platform/layout-dependent algorithm: define accuracy and reproducibility targets
instead of promising universal bitwise equality. Check integer wraparound in both
local accumulation and the combine stage.

### 5C. Selected shape and indexing operations

Add the minimum reshape, axis movement and basic slicing needed by high-value
workloads. Distinguish returned values from NumPy view/aliasing guarantees. Defer
general gathers/scatters, advanced indexing and mutable views unless evidence
justifies expanding scope.

**Acceptance:** C-host examples execute representative broadcast and axis-reduction
expressions without Python, and results retain the declared semantics under different
storage chunking. Native/API ownership, bounds and temporary-buffer behavior are tested.

## 9. Milestone 6 — graph integration and performance

Lower semantically eligible safer-LazyExpr graph segments to native execution.
Keep eligibility explicit: recognized syntax alone does not prove matching dtype,
diagnostics, grouping or supported operand capabilities.

- Introduce an internal/test execution mode that requires native execution for its
  supported subset; fail tests on NumExpr calls or Python numerical fallback.
- Establish what remains a Python frontend and demonstrate that exported artifact
  execution and native scheduling do not depend on that frontend at deployment.
- Reuse immutable parsed/compiled plans keyed by semantic revision and signatures;
  do not cache operand-dependent results across evaluations.
- Fuse compatible arithmetic/function segments while retaining required rounding,
  diagnostics and reduction boundaries. Do not use unconstrained fast-math merely
  to recover benchmark performance.
- Preserve safe/full deserialization policy, native provenance validation, explicit
  table row/partition semantics and unsupported-kernel rejection before writes.
- Make NumExpr optional in packaging only after auditing imports and all intended
  construction, metadata, execution and persistence routes in a clean installation.
- Keep ordinary backend fallback distinct from a persisted portable artifact:
  portable native execution cannot silently fall back to Python.

Benchmark native direct execution, integrated graph execution and existing controls.
Use NumPy for numerical references and NumExpr/current Blosc2 for performance
comparisons without assuming equivalent semantics on divergent cases.

Include tiny/large arrays, float32/64, integers, mixed types, scalar broadcasting,
noncontiguous inputs, partial reads, reductions, persistence and repeated kernels.
Measure compilation, first run, steady state, threading, buffer sizes and peak RSS
separately. Alternate subprocess order and record actual package/native revisions.
Retain isolated baselines from the safer-LazyExpr experiment.

**Acceptance:** a useful end-to-end subset runs with NumExpr unavailable, backed by
capability tests and measured performance. Agree numerical-workload-specific budgets
after baselining; no arbitrary universal slowdown threshold or unmeasured speedup
claim. Comprehensive allocation bounds remain a separate evidence requirement.

## 10. Milestone 7 — scope decision, semantic freeze and release

1. Review every remaining divergence and unverified matrix entry.
2. Choose implement-now, deliberate divergence or unsupported/deferred capability.
3. Freeze type/scalar/operator/function/reduction rules and native diagnostics.
4. Freeze artifact semantic identification and capability negotiation.
5. Verify native interpreter, eligible acceleration, Python integration, persistence,
   builds with optional facilities disabled, and supported platform/wheel/WASM jobs.
6. Publish the compatibility matrix, divergences, standalone C example, performance
   results and migration notes for development artifacts.
7. Release 1.0 when that declared subset is complete; do not wait indefinitely for
   complex numbers, all NumPy APIs or bit-identical host math libraries.

### Pre-release artifact evolution

Draft artifacts already exist despite 1.0 being unreleased. Do not reuse an unchanged
semantic identifier for incompatible arithmetic and silently reinterpret them.
Before the first incompatible change, choose a draft revision marker, supported
version transition or explicit rejection policy. Record it in native validation and
Python import tests. A release need not support every historical draft, but must
identify unsupported semantics rather than execute them under different rules.

**Release acceptance:** all promised capabilities are tested; remaining divergences
are explicit; no required CI failure is hidden by capability skips; examples run
without Python in native hosts; the published claim describes an implemented subset.

## 11. Divergence register and stopping criteria

Maintain a single reviewed register with columns:

| Field | Required content |
| --- | --- |
| ID / operation | Stable identifier, signature and dtype/input category |
| NumPy reference | Pinned version, expected behavior and platform qualifications |
| Current behavior | Actual native result, rejection or missing implementation |
| Category | Value, type, shape, diagnostics, storage, performance or capability |
| Reproducer | Minimal input and conformance case IDs |
| Impact | Relevant workloads and frequency/importance |
| Cost | Implementation, portability, runtime and long-term maintenance cost |
| Decision | Fix for 1.0, deliberate divergence, defer or investigate |
| Evidence | Native/Python revisions, tested platforms and remaining gaps |

Reassess scope after each milestone. Prefer fixing common arithmetic and promotion
errors over reproducing obscure Python-facing details. Defer behavior that depends
on arbitrary Python objects, unstable platform accidents or disproportionate new
runtime machinery. Document deliberate differences as contracts, not apologetic
footnotes. The register is a release artifact, not a permanently growing hidden todo.

## 12. Verification and working discipline

- Native and Python tests must exercise the same declared semantic revision.
- Test actual NumPy/native versions and package paths; CI labels are not evidence.
- Cover Linux, Windows and macOS plus 32-bit WASM behavior. Test NumPy 1.26 host
  support separately from the pinned NumPy 2.x semantic reference.
- Run portable interpreter-only builds and eligible JIT/SIMD variants; record
  capability skips and pre-execution fallback explicitly.
- Use native sanitizers for arithmetic UB, bounds and lifetime errors where
  supported. Stress/sanitizer success is evidence, not a general safety proof.
- Keep ordinary GIL, free-threaded builds and actual no-GIL execution distinct.
  Numerical compatibility adds no automatic same-storage mutation guarantee.
- Run Python/build/install commands in the `blosc2` conda environment. Follow each
  repository's AGENTS.md and existing C/Cython/Python style and build conventions.
- Add meaningful tests with each behavioral change; run focused checks first and
  full native/Python suites at integration checkpoints. Record exact paired SHAs
  and local changes included in a result.
- Do not change defaults, add dependencies or publish changes merely because this
  plan describes future integration. Those actions follow their normal explicit
  project decisions and permissions.

## 13. First implementation checkpoint

The first work unit should complete milestone 1 and a small end-to-end slice of 2:

1. Record both branch starting SHAs and confirm NumPy 2.5.3 as the proposed reference.
2. Inventory existing native type/cast rules and create the initial divergence table.
3. Define the vector schema and add representative cases for integer boundaries,
   float32 literals, signed/unsigned promotion, division/remainder and explicit casts.
4. Run them through NumPy generation, native interpreter and Python integration,
   preserving current failures as classified evidence rather than changing semantics
   and the reference format simultaneously.
5. Select the first milestone-3 vertical slice from measured impact and difficulty;
   fixed-width integer arithmetic and scalar promotion are leading candidates.
6. Approve semantic/artifact revision handling before that first incompatible change.

Thereafter, iterate in small cross-repository slices. Functions, array scheduling
and performance can progress incrementally once their type/diagnostic prerequisites
are stable; there is no need to finish every possible operation in one layer before
demonstrating a useful native workload in the next.
