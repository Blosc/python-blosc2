# Portable DSL 1.0 implementation plan

Date: 2026-10-07. Status: implementation plan; all three detail decisions
below are confirmed by the user. Feature scope and earlier semantic decisions are
approved in `portable-DSL-1.0-scope-inventory.md`. This file intentionally uses
the requested filename.

## Objective and shortest implementation path

Deliver a public, versioned kernel contract executable from C and Python, with
portable-only new kernel persistence. Build correctness in the native
**interpreter first**. Reuse existing parsing, expression evaluation, strings,
ND support and artifact loading; do not implement a second language engine.

Initially route every 1.0 kernel through the interpreter. Subsequently enable
only existing JIT paths whose semantics are demonstrated compatible. Missing
JIT support, including checked arithmetic, masked reductions or strings, is
never a reason to delay interpreter delivery or weaken the contract.

Draft portable 1.0 is the sole public portable format and the authoring/validation
default. The unreleased 0.1 profile is retired, with explicit unsupported-version
rejection rather than migration or reinterpretation. Ordinary full-DSL
execution keeps its existing semantics. Miniexpr owns normative semantics,
validation and conformance; Python-Blosc2 owns normalization and host integration.

## 0. Confirmed behavioral choices

The user approved Q1–Q3 below. Their recommended rules are now implementation
requirements; the explanations preserve the rationale for these choices.

### Q1. Complete the numeric type policy

Recommend Boolean, signed/unsigned 8/16/32/64-bit integers, and float32/float64.
No float16, complex, object, datetime or structured arithmetic in 1.0.

Proposed promotion rules:

- Same typed operands retain their type; Boolean arithmetic uses int64 numeric
  zero/one, with `/` producing float64. Boolean logical operations remain Boolean.
- Same-signedness integers widen to the wider operand. Mixed signed/unsigned
  integers use the smallest signed type covering both complete operand ranges.
  Reject an implicit combination for which no such type exists, notably
  int64/uint64; explicit float conversion remains possible with documented loss.
- float32 with float32 stays float32; float64 dominates. Integer/float mixtures
  use float32 for integers of at most 16 bits and float64 otherwise. Conversion
  to float64 may round 64-bit integers; document this rather than claiming exactness.
- Weak source literals adopt an accompanying typed operand when within range.
  Decimal floating literals round once to the selected IEEE type using nearest,
  ties-to-even; do not require exact binary representability of `0.1`.
  An out-of-range contextual literal is rejected, requiring an explicit cast or
  typed constant. Literal-only integers default to int64 and floats to float64.
- Typed captured constants are strong operands, not weak literals. `int()` yields
  int64, `float()` yields float64, and `bool()` yields Boolean. Explicit typed
  casts, if admitted in the builtin matrix, have precisely named destination types.
- Arithmetic promotion does not define comparisons by accident: compare mixed
  signed/unsigned integers exactly without lossy float intermediates. Reject any
  unsupported mixed comparison explicitly in the initial matrix.

Why approval matters: uint64 is useful, but there is no exact fixed-width signed
common type with int64. This policy chooses a diagnostic rather than implicit
float64 rounding for their arithmetic. Narrow integer arithmetic can overflow
before final widening, consistently with operand-driven computation.

### Q2. Represent block-scalar outputs in the lazy host

Recommend explicit result cardinality in the artifact: `elementwise` or
`block_scalar`. A standalone C block evaluation returns one value for the latter.
In Python lazy execution, expose one value per logical evaluation block, on the
block-grid shape, with row-major block ordering. Do not broadcast an actual
block-scalar return back to every element implicitly.

A scalar reduction used inside an elementwise calculation broadcasts to its
participating lanes; that does not itself make the kernel block-scalar.
Reject paths with inconsistent return cardinality. A block-scalar partial read
indexes output blocks; an elementwise partial read evaluates original input
groups before selecting lanes. Empty input arrays have no scheduled blocks;
an explicitly evaluated empty standalone block still obeys reduction identities.

Why approval matters: today's artifact API assumes `nitems` output values.
Silently keeping that allocation model would confuse scalar reductions with
elementwise results and obscure what a saved array's shape means.

### Q3. Reuse the existing legacy opt-in

Recommend user-supplied `deserialize="full"` as the legacy-kernel opt-in, rather
than adding another public switch. Normal/default safe opens must not reconstruct
legacy Python kernels. Propagate the effective policy through nested operands;
internal calls that hard-code `full` must not grant permission on the user's
behalf. If existing API defaults cannot distinguish explicit permission from an
implicit default, adjust the boundary/policy propagation before admitting legacy
execution. Portable artifacts need native validation, not Python reconstruction.

No migration tool, automatic conversion, or legacy-save fallback is planned.
Manual adaptation guidelines are the only conversion assistance.

## 1. Establish a small normative contract and baseline

Files: `../miniexpr/doc/dsl-spec/1.0.md`, `artifact-1.0.md`, a machine-readable
feature/type/function matrix, and `../miniexpr/tests/portable-dsl/` fixtures.

1. Retire unreleased 0.1 validation/loading and obsolete freeze documents. Retarget
   useful encoding/numeric/masking/validation fixtures to draft 1.0, updating
   independent references for operand typing and checked arithmetic.
2. Inventory actual builtin registrations and interpreter implementations once.
   Classify each operation as included, existing full-DSL only, or excluded;
   include arity, type combinations, result type, errors and accuracy policy.
   Avoid another unbounded backend audit. Review omissions of useful existing
   operations before freezing the matrix.
3. Cover arithmetic, comparisons, bitwise operations, shifts, casts, `where`,
   control flow, supported math, fixed-width string operations, ND symbols and
   the existing numeric/Boolean reduction families. Do not assume scalar and
   reducing forms of similarly named functions share semantics.
4. Spell out floor division/remainder signs, integer power and shift overflow,
   signed-zero behavior, NaNs, infinities, local type joins, short-circuiting and
   evaluation order. Recommend typed locals with a statically determined join
   type and rejection when no supported join exists; no dynamic dtype changes.
5. Specify loop caps and missing-return errors. Resource caps are host policy;
   an error must not become a partial successful result. Preserve existing
   configured caps rather than inventing a new execution sandbox.
6. Use semantic capability IDs for ND context, fixed strings and block reductions.
   A declared capability is a requirement, not permission to skip validation.

Exit: a finite matrix and example fixtures cover every admitted operation;
no undefined implementation behavior is advertised as portable semantics.

## 2. Introduce versioned semantic compilation and conservative dispatch

Primary files: native `dsl_portable.c`, `dsl_compile.c`,
`dsl_compile_internal.h`, `dsl_compile_support.c`, `dsl_artifact.c`.

1. Thread a portable semantic profile through compilation and compiled program
   metadata. Keep it out of process-global mutable switches.
2. Validate normalized source, resolved call targets, compiled node types,
   effects, return cardinality and inferred capability requirements. Use both
   source and typed-tree gates.
3. Reject callbacks, complex values, `print`, fast/contract FP, dynamic widths,
   arbitrary memory access and unsupported reduction placements. A reduction
   condition inside a loop is not a reduction assignment in the loop body.
4. Initially mark every 1.0 program interpreter-only. Optional JIT requests fall
   back before compilation/execution of an incompatible backend. Explicit
   required-JIT requests fail clearly when no certified path exists.
5. Preserve source locations through compilation and expose stable unsupported,
   binding, format and evaluation error categories.

Exit: 1.0 kernels cannot accidentally execute
under full-DSL output-context typing or an uncertified JIT.

## 3. Implement checked operand-typed numeric interpretation

Primary files: `dsl_compile.c`, `dsl_eval.c` and the shared native expression
typing/evaluation helpers they call. Locate shared helpers before changing them;
make changes profile-aware where ordinary DSL semantics differ.

1. Infer every expression's computation dtype independently of declared output.
   Preserve weak-literal metadata until contextual typing is complete. Insert
   explicit typed conversion nodes rather than relying on output buffers.
2. Implement the approved promotion table and casts. Round float32 intermediates
   at operation boundaries; prohibit implicit contraction/reassociation in the
   strict profile. Test that changing output dtype does not change intermediates.
3. Implement checked integer add/subtract/multiply/negation, division/remainder,
   shifts, powers and all narrowing/conversion paths actually admitted by the
   matrix. Check before performing a C operation that could overflow or be UB.
   Do not rely on signed overflow, invalid shifts or out-of-range C casts.
4. Check float-to-integer limits without rounded-boundary errors, including
   int64/uint64 extrema. Fractional values truncate toward zero before range
   acceptance is decided. Cover exact large integers without double round trips.
5. Apply numeric-zero/one Boolean arithmetic consistently to intermediates,
   explicit casts and final output. Preserve fractional division before truth.
6. Propagate failures from nested expressions and final conversions. Only active
   lanes execute checked operations. Short-circuit branches and `where` must not
   report failures from unselected lanes; implement masked operand evaluation
   rather than evaluating invalid discarded branches eagerly.
7. Document floating math domains and per-function tolerances. Use a bounded
   cross-platform accuracy contract informed by existing libm implementations;
   do not make bitwise transcendental equality a release requirement.
8. Keep workspace and error state per evaluation. On failure, native output is
   unspecified and must not be consumed; Python raises rather than caching or
   returning that block as a successful result.

Exit: exact integer/error cases, Boolean fractions, float32 rounding, mixed
types and inactive invalid branches pass interpreter conformance on native and
WASM targets. Additional JIT arithmetic work is unnecessary for this milestone.

## 4. Make masking and limited block reductions explicit

Primary files: `dsl_eval.c`, `dsl_eval_internal.h`, `dsl_compile.c`,
`miniexpr_eval_reduce.c`, and ND evaluation adapters.

1. Represent valid lanes and currently participating lanes explicitly. Track
   branch, loop, break, continue and return masks; nested conditions reduce over
   the mask participating at that program point. Continue excludes the rest of
   that iteration but does not permanently remove a lane from the next one.
2. Top-level reductions see valid block elements; nested supported conditions see
   active valid elements. Never include padded lanes or stale local storage.
3. Retain the restriction on reduction-valued assignments/returns in control-flow
   bodies. Admit top-level reductions and supported conditions, including nested
   `if all(...)` in Mandelbrot. Validate placements structurally, not by a string
   search or a blanket ban on nonzero nesting depth.
4. Accumulate in logical C-order within a block; skip inactive lanes without
   changing the order of remaining lanes. Disable parallel/tree reduction and
   fast reduction shortcuts for this profile unless they give the required order.
5. Recommended operation details for the normative matrix: widen signed/Boolean
   sum/prod to int64 and unsigned to uint64; keep floating sum/prod input dtype;
   integer mean uses float64; floating mean uses its floating input dtype.
   Checked integer accumulation applies at every step. Mean uses the specified
   ordered sum followed by division by the participating count.
6. Use approved empty identities for any/all/sum; recommend prod(empty)=1 and
   mean(empty)=NaN. Empty min/max errors. Recommend NaN propagation for numeric
   reductions, deterministic signed-zero ties for min/max, and NaN-as-true for
   truth reductions, consistently with Boolean conversion. Record these rules
   explicitly and compare against existing behavior before implementation.
7. Distinguish reduced scalars from per-lane values in compiled metadata and
   allocation. Broadcast scalar operands only onto participating lanes.

Exit: Mandelbrot interpreter results survive changes in partition when its
early-exit optimization is semantically neutral; a deliberately block-dependent
example changes exactly as specified. Include partial masks, edge padding,
empty groups, continue/break/return, ordered cancellation and overflow cases.

## 5. Extend native artifacts for context, widths and cardinality

Primary files: `dsl_artifact.c`, `miniexpr_artifact.h`, `dsl_portable.c`.

1. Keep standalone UTF-8 JSON, explicit schema/language versions, strict semantic
   requirements, named signatures, exact typed constants and optional bindings.
   Extend integer encodings for the admitted widths; retain decimal integer
   strings, IEEE floating hex and Boolean values.
2. Add string descriptors with code-unit kind and fixed width. Recommend canonical
   hex bytes for bytes constants and fixed-endian 32-bit code-unit hex for Unicode
   constants, preserving values without assuming all slots are valid UTF-8 text.
   Native buffers use host representation; adapters perform endian conversion.
3. Add result cardinality and queries for itemsize/output capacity. Introduce an
   extended evaluation descriptor/API rather than silently changing the old
   `me_artifact_eval()` ABI, whose output currently has `nitems` elements.
4. The extended call supplies logical shape, block origin/extent, valid-lane
   description and explicit output capacity. Define `_flat_idx` in the logical
   domain, `_i<d>` as coordinates in that domain and `_n<d>` as logical sizes.
   Broadcasting inputs does not change the output coordinate domain.
5. Validate required context, dimensions, products, byte capacities, integer
   conversions, ownership and overlap before evaluating. No array-sized constant
   replication; no mutable context retained on a shared handle.
6. Distinguish zero-element elementwise calls from explicitly empty block-scalar
   reduction calls; the latter must execute empty-group semantics.
7. Validate inferred capabilities against artifact declarations. Unknown versions
   or unavailable capabilities fail explicitly, without Python reconstruction.

Exit: a standalone C runner loads each capability fixture and evaluates supplied
buffers/context using only native code; incorrect descriptors fail deterministically.

## 6. Admit existing fixed-width strings and logical ND kernels

Native files: existing string implementation, `miniexpr_eval_dsl_nd.c`,
`miniexpr_eval_nd.c`, plus compile/validation and artifact descriptors above.

1. Reuse native string operations and width inference. Specify first-NUL behavior,
   full-width unterminated slots, no Unicode/bytes mixing, bytes comparison rules,
   Unicode code-unit order, case-expansion truncation and static width bounds.
2. Pin the Unicode mapping/whitespace data semantics used by admitted operations;
   avoid locale-dependent behavior. Separate arbitrary byte constants from the
   current restriction on non-ASCII source literals used against bytes operands.
3. Check branch/local width joins and allocate the inferred return width, not a
   NumPy approximation. Reject unbounded results before execution. Keep strings
   interpreter-only; a host may pack fixed-width output into varlen storage.
4. Define sliced/view authoring domains explicitly: a newly constructed kernel
   over a view has its declared logical domain; reading a slice of an existing
   saved kernel retains that kernel's original coordinates and reduction groups.
5. Exercise zero-input constructors using ND context, sliced domains, broadcasting
   and padded edge blocks. Pass context through native evaluation, not synthetic
   Python callbacks or hidden global arrays.

Exit: representative existing string/ND kernels export and execute identically
in C and Python, including widths and coordinates, with no JIT dependency.

## 7. Python bridge, normalization and lazy execution

Files: `src/blosc2/portable_kernel.py`, `dsl_kernel.py`, `blosc2_ext.pyx` and
associated native declarations, `lazyexpr.py`.

1. Make export/validation APIs default to draft 1.0; reject retired versions.
   select 1.0 explicitly for new persistence until the public-default transition
   is documented. Import remains native-only.
2. Reuse NumPy-call, string-syntax and static-row normalization; record named
   column bindings and snapshot supported typed constants, including strings.
   Reject opaque captures without invoking arbitrary conversion hooks.
3. Pass context, width, shape/cardinality and evaluation errors through Cython.
   Extend `PortableKernel.evaluate()` accordingly. Build interpreter-backed lazy
   execution around the native handle, not around `kernel_from_source()` or exec.
4. Define a persisted logical block grid with shape, block extents, domain origin
   and C-order traversal. Bind it once for reduction-bearing kernels; do not
   derive it anew from current physical chunks at each evaluation.
5. Schedule complete original reduction groups for elementwise partial reads;
   gather/scatter across physical storage as necessary. Elementwise-only kernels
   may be tiled freely while retaining original logical ND coordinates.
6. Implement the approved block-scalar output mapping and distinguish input
   domain shape from result shape. Validate immutable bound dtypes/widths and
   domain requirements when reopening referenced operands.
7. Preserve author-written compiler pragmas as compiler selection preferences;
   local defaults must not override them. They do not exempt the backend from
   portable semantic checks or disable the agreed interpreter fallback.

Exit: ND/string/reduction kernels run lazily through native handles, with slice
results and reduction grouping unchanged after storage rechunking.

## 8. Enforce portable-only kernel persistence end to end

Files: `b2objects.py`, `lazyexpr.py`, `deserialization.py`, relevant open/from-frame
entry points and CTable computed/generated-column serializers.

1. Trace every DSL-kernel save/load route: structured B2Objects, older LazyUDF
   metadata paths, embedded/nested lazy operands, CTable columns and frames.
   Use one encoder/validator and one native decoder rather than divergent formats.
2. Preflight normalized kernel portability and typed bindings before creating or
   replacing the destination. Save the artifact plus explicit operand references,
   logical context/partition and container format discriminator.
3. Remove broad exception-driven fallback to source-based persistence for DSL
   kernels. A portability failure must survive to the caller with location/reason.
   Do not redefine ordinary non-kernel LazyExpr persistence by accident.
4. On load, distinguish portable, legacy and malformed records using metadata
   only. Validate portable artifacts natively before binding/execution; never
   reconstruct Python from a portable record, even after validation fails.
5. Apply explicit legacy opt-in consistently, including recursive operands.
   Review current hard-coded `deserialize="full"` calls in `b2objects.py` and
   effective defaults in `deserialization.py`; do not allow an internal reopen
   to bypass the user's selected policy.
6. No automatic migration, conversion command or source-rewrite-on-open.
   Document manual changes and explicit legacy use. Container/data-source
   portability is separate from standalone kernel portability; demonstrate C
   execution of the stored artifact with supplied bindings, without promising
   C support for every Python remote/container protocol.

Exit: all new DSL save routes reject excluded features; round trips preserve
native-only execution and block context; legacy files require opt-in everywhere.

## 9. Optional JIT eligibility, not a JIT implementation project

After interpreter milestones, build a conservative eligibility predicate over
the typed program and backend capabilities. Its default answer is no.

- Allow only operation/type/context combinations already shown to implement 1.0
  semantics. Include semantic version, dtype/width and relevant policy in cache
  identity so cached full-DSL programs cannot masquerade as 1.0 programs.
- Keep checked integers, unsupported casts, strings, ND/reduction combinations
  and uncertain masking on the interpreter until individually certified.
- Optional JIT unavailability or unsupported lowering selects interpreter before
  execution. Do not retry an executed, failing kernel using another backend;
  arithmetic/domain failures remain failures.
- Required-JIT remains an explicit host diagnostic mode: report unsupported
  acceleration instead of falsely claiming JIT execution.
- Retain the Windows interpreter baseline and WASM exact-integer fallback.
  No target is required to acquire a new code generator for 1.0.

Exit: eligible kernels match interpreter semantics; ineligible kernels execute
correctly by default and report a reason when JIT is explicitly required.

## 10. Bounded conformance and integration verification

Do not cross-product dtype, operation, backend, length and storage dimensions.
Put the detailed semantic corpus in miniexpr and keep approximately 100–200
orthogonal Python portable tests, restructuring existing cases as necessary.

Native fixtures cover exact boundary/error cases for each checked operation;
representative promotion/literal pairs; rounding and inactive-lane behavior;
Mandelbrot and masked reductions; ND origins/padding; string encodings/widths;
artifact errors, capacity and ownership; immutable concurrent handles.

Python integration covers normalization and captures, standalone-native import,
native error propagation, each save route, legacy opt-in, rejection without
partial destination replacement, partial reads/rechunking, scalar-block shapes,
default fallback and explicit required-JIT failure. Use exact expected values
for integers rather than a NumPy overflow reference. Floating expectations use
the specified rounding/order or function-specific tolerance.

Normal Python tests use the published/pinned miniexpr and self-contained fixtures.
Keep external corpus/C-runner checks opt-in through the established environment
variables; never infer a sibling checkout at runtime.

Run targeted tests after each milestone, native sanitizers for masks/capacity/
arithmetic, then full native/Python CI including Windows and WASM at integration.
Use the `blosc2` conda environment for Python, builds and tests. Smoke-test an
isolated wheel built without local native overrides. Test artifact-disabled
builds report missing capability cleanly. Update the published native pin only
after the required native code is available; no unpublished-checkout dependency.

## Delivery order and completion checklist

### Autonomous execution and durable handoff

Follow the milestones in order, recording progress and verification after each.
The autonomous implementation target is completion of the beta-readiness gates,
not collection of external beta feedback or automatic freezing of the contract.

- Repository scope covers the necessary changes in Python-Blosc2 and miniexpr.
  Follow each repository's local instructions. Preserve unrelated user work;
  avoid unrelated refactors and whole-language JIT audits.
- Prioritize interpreter correctness and use pre-execution JIT fallback. Do not
  spend a milestone's effort implementing optional acceleration.
- Continue through routine implementation, build and test failures without
  asking permission. Resolve ordinary details from approved semantics, verified
  implementation behavior and the normative matrix.
- Pause only for unresolved semantic decisions, scope contradictions or external
  blockers such as unavailable credentials. Record the precise blocker and
  proposed resolution; continue independent unblocked milestones where possible.
- Commits, pushes, publishing and other repository/release state changes require
  explicit user authorization consistent with repository instructions. This plan
  does not itself grant those permissions. Run available local verification;
  record any CI checks blocked by missing authorization or credentials.
- Keep a durable handoff in `plans/portable-DSL-1.0-progress.md`. Create it at the
  start of implementation and update it after every milestone and before any
  interruption or context handoff. Include completed checklist items, changed
  files in both repositories, exact verification commands/results, unresolved
  failures, decisions and their approval status, and the next concrete action.
  Record relevant native revision/build/pin information so another session can
  reproduce the environment. Do not store credentials or secrets.
- On resume, read this plan, the scope agreement, the progress file and current
  working-tree changes before editing. Do not assume an interrupted command
  completed or discard partial work. Reverify uncertain results.
- At completion, summarize delivered behavior and remaining external release
  gates. Do not claim CI success, beta feedback or release publication without
  evidence.

Checklist status is **local implementation**, not release/platform certification;
authoritative commands/results are in `portable-DSL-1.0-progress.md`.

1. [x] Finish finite normative draft type/arity/domain/tie/NaN/reduction/Unicode
   matrix (Q1–Q3 confirmed), with analytic/exact-reference fixtures. Independent
   function-specific cross-platform libm accuracy bounds remain a release gate.
2. [x] Add versioned compilation/interpreter-only 1.0 dispatch, including public
   core-only source validation and extended ND/cardinality/string-width descriptor.
3. [x] Complete checked operand-typed numeric semantics and active-lane errors.
4. [x] Complete reduction masks/cardinality, including rejection of divergent
   scalar return sites and preservation of finished-lane iterator values.
5. [x] Extend artifact ABI/schema/context and native runner/descriptor fixtures.
6. [x] Admit bounded fixed strings and logical ND descriptors through native code.
7. [x] Integrate Python native handles and immutable logical-block scheduling.
8. [x] Enforce validated saves/native loads and recursive explicit legacy opt-in
   across the implemented disk/frame/container/table routes, with safe MessagePack
   defaults. No source fallback; table row-domain modes are explicit.
   New authored DSLKernel-backed LazyUDF saves automatically normalize and export
   validated 1.0 artifacts, with typed scalar/capture snapshots and named array
   bindings. Preserve the elementwise domain and current uniform logical block
   grid; reject true block-scalar kernels with explicit portable API guidance.
   This is authoring preflight, not automatic migration of loaded legacy carriers.
   Explicit independent-row table APIs accept DSLKernel authoring plus output
   dtype/cardinality directly; old vector-column APIs have no row-domain contract
   and reject persistence with a specific ambiguity diagnostic.
9. [x] Conservative interpreter-only dispatch and pre-execution optional JIT
   fallback; required-JIT rejects unsupported acceleration explicitly.
10. [x] Complete release verification gates for the tested platform matrix:
    - [x] Compact local Python/native conformance and full local ASAN/UBSAN.
    - [x] Isolated native-development override wheels and artifact-disabled smoke.
    - [x] Draft/API docs build; changed portable page free of diagnostics.
    - [x] Native Windows/Linux/macOS/WASM conformance matrix (all seven jobs green).
    - [x] Independent finite math corpus and stock-pin macOS execution: 284 samples,
      40 canonical operations, both floating precisions, function-specific budgets.
    - [x] Execute the independent math corpus on remaining Python release platforms
      (no universal ULP bound inferred from the finite corpus or host libm).
    - [x] Publish/pin the native revision and verify a stock-pin macOS ARM64 wheel.
    - [x] Complete the remaining stock-pin platform wheel matrix (27 successful jobs;
      retain the existing untested macOS Intel cp315/cp315t dependency exception).
    - [x] Whole-tree docs disposition: fix parsing errors, exclude generated input
      trees, and inventory/defer existing image/reference warnings outside Menudet.

Each step must land as a usable tested increment; reduction and string fixtures
can be prepared while numeric interpretation is implemented, without introducing
unapproved semantics. If a useful existing feature cannot fit the finite matrix,
raise the specific incompatibility rather than silently shrinking scope.

Release documentation includes normative native specs, Python API/persistence
guidelines, manual legacy adaptation, standalone C examples, and corrected DSL
reference statements. These checks establish readiness for the beta milestone
below; freezing the public 1.0 contract additionally requires beta feedback.
Do not tie completion to broad JIT acceleration, migration utilities or new
remote-storage protocols.

## Release milestone: miniexpr beta and portable 1.0 release candidate

Once the implementation checklist passes, move miniexpr into beta with portable
DSL 1.0 as a **release-candidate specification**. Interpreter conformance across
supported platforms, native artifact loading, and working persistence integration
are the entry gates. Limited JIT coverage does not block beta: correct
pre-execution interpreter fallback is sufficient.

Keep the two release promises distinct:

- **Miniexpr beta:** implementation and APIs are approaching release readiness;
  broader use should exercise integration, usability and performance.
- **Portable DSL 1.0, once frozen:** a public compatibility commitment independent
  of whether the miniexpr implementation is still in beta.

During beta, exercise real persisted kernels and cross-language C/Python loading,
including ND context, fixed-width strings, block reductions, partial reads and
legacy opt-in. Resolve contract ambiguities and compatibility issues found in
that feedback before freezing 1.0. Clearly label release-candidate artifacts and
their compatibility status; do not present draft artifacts as already covered
by the final 1.0 guarantee.

- [ ] Enter miniexpr beta after the implementation and verification gates pass.
- [ ] Publish portable 1.0 as a release-candidate specification during beta.
- [ ] Review real-world persistence and cross-language feedback; resolve blockers.
- [ ] Freeze and publish the portable 1.0 compatibility contract after that review.

Further JIT acceleration remains an independent optimization, not a prerequisite
for either beta entry or contract freeze.
